import matplotlib
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import gymnasium as gym
from fw.collision_avoidance.scenario_builder import ScenarioBuilder
from fw.collision_avoidance.scenario_classifier import ScenarioClassifier, NUM_SCENARIOS
from scipy.stats import gaussian_kde

import math

from fw.environments.py_sim_env import PySimEnv

# First try interactive backend
try:
    matplotlib.use('TkAgg')  # Try using the interactive backend
except ImportError:
    print("Warning: 'TkAgg' backend not available. Falling back to 'Agg'.")
    matplotlib.use('Agg')  # Use non-interactive backend as fallback

from typing import Optional
from fw.ships.ship_factory import ShipModel
from fw.dynamics.dynamics_model import Model
from typing import Tuple, Dict
import numpy as np


# -----------------------
# Module-level constants
# -----------------------
# Physical limits / config

# Grid limits
CHECKPOINTS_DISTANCE = 350
MIN_GRID_POS = -11700
MAX_GRID_POS = 14500

# Reward / shaping
SUCCESS_REWARD = 50.0
CHECKPOINT_AREA_SIZE = 5.0
TARGET_AREA_SIZE = 7.0

# Rendering constants
MAX_FIG_WIDTH = 1200
MAX_FIG_HEIGHT = 900
DPI = 100

# Constants definition
PERPENDICULAR_LINE_LENGTH = 50.0
PERPENDICULAR_LINE_PROXIMITY = 2.0
HEADING_CHANGE_THRESHOLD = np.pi / 2.0
CROSS_TRACK_TERMINATION_MULTIPLIER = 2.0
EPSILON = 1e-9  # Small epsilon for numerical stability


class PySimColavEnv(PySimEnv):
    """Custom Python Simulator environment for ship navigation with improved physics."""

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
            enable_obstacles: bool = True,
            use_polar_scan: bool = True,
            enable_variable_ship_size: bool = False,
            eval: bool = True,
            eval_scenario: Optional[str] = "crossing-right", # "static", "overtaking", "head-on", "crossing-right", "crossing-left"
            eval_obstacle_count: int = 1,
            include_scenario_one_hot: bool = False,
    ) -> None:
        self._first_class = False
        self.first_class = False

        # Scenario measurements
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
        self.obstacle_population = [0, 1, 2, 3]
        self.obstacle_weights = [0, 100, 0, 0]
        self.obstacle_probs = np.asarray(self.obstacle_weights, dtype=np.float32)
        self.obstacle_probs /= self.obstacle_probs.sum()
        self.static_ship_lateral_max = 100.0

        # OS and TS speed
        self.os_nominal_speed = 2.0
        self.overtaking_ts_v_min = 1.0
        self.overtaking_ts_v_max = self.os_nominal_speed - 0.3
        self.headon_ts_v_min = self.os_nominal_speed + 1.0
        self.headon_ts_v_max = self.os_nominal_speed + 3.0
        self.crossing_ts_v_min = 1.0
        self.crossing_ts_v_max = 5.0
        self.crossing_activation_margin = 0.0  # extra margin beyond polar_max_range
        self.crossing_speed_factor = 1.0  # > 1.0 makes TS arrive before OS, forcing reaction
        self.own_ship_radius = 0.0

        # polar scan
        self.use_polar_scan = bool(use_polar_scan)
        self.polar_scan_cached = None
        self.polar_n_sectors = 54
        self.polar_fov = np.radians(270)
        self.polar_max_range = 500.0
        self.polar_obs_max = 0.1

        self.eval_obstacle_count = int(eval_obstacle_count)
        self.eval_scenario = str(eval_scenario)
        self.eval = bool(eval)
        self.include_scenario_one_hot = bool(include_scenario_one_hot)
        self.min_distance = float("inf")

        # Per-episode minimum distance tracking for evaluation
        self._episode_min_distances: list[float] = []  # completed non-collision episodes
        self._current_episode_min_dist: float = float("inf")
        self._current_episode_collided: bool = False

        self._last_collision_info = {}
        self._colreg_violations = 0
        self._colreg_correct = 0
        self._colreg_initial_heading = None
        self._crossing_left_initial_speed = None
        self._crossing_right_grav_point = None

        # Passing snapshot: list of dicts with OS/TS positions at the moment of passing
        self._passing_snapshots: list[dict] = []

        bin_width = float(self.polar_fov) / float(self.polar_n_sectors)
        self._polar_rel_angles = (
                -(self.polar_fov / 2.0) + (np.arange(self.polar_n_sectors, dtype=np.float32) + 0.5) * bin_width
        ).astype(np.float32)
        self._polar_rel_dirs = np.stack(
            (np.cos(self._polar_rel_angles), np.sin(self._polar_rel_angles)), axis=1
        ).astype(np.float32)

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

        self.polar_scan_cached = None

        self.scenario_builder = ScenarioBuilder(self)
        self.scenario_builder.cache_checkpoint_segments()

        # Scenario classifier for the single OS
        self.scenario_classifier = ScenarioClassifier(
            classify_range=self.polar_max_range,
            forget_range=self.polar_max_range + 20)
        # Track which TS index the one-time COLREG reward was already given for (None = not given)
        self.ts_passed = None

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

        self.scenario_builder.sample_collision_entity_scenario()
        if self.use_polar_scan:
            self._rebuild_polar_scan_geometries()

        # important: reinitialize after scenario sampling
        self._initialize_simulation()

        self.reward_weights = {
            "polar_distance": 0.7,
            "colreg_bearing": 0.4,
            "headon_bearing": 0.4,
            "crossing_left_speed": 0.3,
            "crossing_right": 0.5,
            "crossing_right_grav": 1.0,
        }

        # CRI parameters (Zhen et al. 2022 / Zhao et al. 2026)
        # k1, k2: peak risk value at DCPA=0 / TCPA=0 (scale of each component)
        # alpha1, alpha2: negative decay rates (more negative = risk drops faster with distance/time)
        # D: safe distance threshold [m] — CRI_DCPA = k1 * exp(alpha1 * DCPA/D)
        # T: safe time threshold [s]   — CRI_TCPA = k2 * exp(alpha2 * TCPA/T)
        # omega1, omega2: weights in the weighted sum RI = omega1*CRI_DCPA + omega2*CRI_TCPA
        self.cri_k1 = 1.0
        self.cri_k2 = 1.0
        self.cri_alpha1 = -1.0       # spatial decay rate
        self.cri_alpha2 = -1.0       # temporal decay rate
        self.cri_D = 50.0           # safe distance [m]
        self.cri_T = 120.0           # safe time horizon [s]
        self.cri_omega1 = 0.5        # weight for CRI_DCPA
        self.cri_omega2 = 0.5        # weight for CRI_TCPA

        self._collision_entity_patches = []
        self._collision_entity_render_key = None

    # -------------------------
    # Utility & Math
    # -------------------------
    def _compute_cri(self, os_pos, os_heading, os_u, os_v, ts) -> float:
        """
        Compute Collision Risk Index (Zhen et al. 2022 / Zhao et al. 2026).
        Returns RI_collision in [0, k1*omega1 + k2*omega2] (typically [0, 1]).
        """
        ts_pos = ts.get_position()
        ts_vel = ts.get_velocity()  # [vx, vy] in world frame

        # OS velocity in world frame (body-to-world transform)
        cos_h = math.cos(os_heading)
        sin_h = math.sin(os_heading)
        os_vx = os_u * cos_h - os_v * sin_h
        os_vy = os_u * sin_h + os_v * cos_h

        # Relative position and velocity (TS relative to OS)
        dx = float(ts_pos[0] - os_pos[0])
        dy = float(ts_pos[1] - os_pos[1])
        dvx = float(ts_vel[0]) - os_vx
        dvy = float(ts_vel[1]) - os_vy

        vr_sq = dvx * dvx + dvy * dvy
        if vr_sq < 1e-12:
            return 0.0  # no relative motion, no convergence risk

        vr = math.sqrt(vr_sq)

        # TCPA and DCPA via vector projection
        tcpa = -(dx * dvx + dy * dvy) / vr_sq
        dcpa_x = dx + dvx * tcpa
        dcpa_y = dy + dvy * tcpa
        dcpa = math.sqrt(dcpa_x * dcpa_x + dcpa_y * dcpa_y)

        # CRI components (exponential decay)
        cri_dcpa = self.cri_k1 * math.exp(self.cri_alpha1 * dcpa / self.cri_D)

        if tcpa <= 0.0:
            cri_tcpa = 0.0  # ships diverging, no temporal risk
        else:
            cri_tcpa = self.cri_k2 * math.exp(self.cri_alpha2 * tcpa / self.cri_T)

        return self.cri_omega1 * cri_dcpa + self.cri_omega2 * cri_tcpa

    def _nearest_segment_signed_cte(self, obs_pos: np.ndarray) -> float:
        """
        Walk forward from the OS's current segment, accumulating path length.
        Stop when accumulated length exceeds polar_max_range. Among the
        segments within that window, return the signed CTE (via cross product)
        of the segment whose perpendicular projection of the obstacle falls
        within the segment bounds (0 <= t <= 1) and has the smallest
        perpendicular distance.

        If no segment has a valid in-bounds projection, falls back to the
        segment with the smallest absolute perpendicular distance overall.

        Returns:
            Signed perpendicular distance. Positive = port of segment
            direction, negative = starboard. The sign flips automatically
            for a ship travelling the reverse path because its segment
            vectors point the other way.
        """
        max_path_dist = self.polar_max_range
        accumulated_len = 0.0

        best_bounded_abs = float('inf')
        best_bounded_signed = 0.0
        best_unbounded_abs = float('inf')
        best_unbounded_signed = 0.0

        start_idx = max(1, self.checkpoint_index)

        for seg_idx in range(start_idx, len(self.checkpoints)):
            seg_start = np.asarray(self.checkpoints[seg_idx - 1]['pos'], dtype=np.float32)
            seg_end = np.asarray(self.checkpoints[seg_idx]['pos'], dtype=np.float32)
            segment_vec = seg_end - seg_start
            seg_len = float(np.linalg.norm(segment_vec))

            if seg_len < EPSILON:
                continue

            # Check if this segment is still within rangefinder reach
            if accumulated_len > max_path_dist:
                break
            accumulated_len += seg_len

            to_obstacle = obs_pos - seg_start
            # Projection parameter: where the obstacle projects onto the segment line
            t = float(np.dot(to_obstacle, segment_vec)) / (seg_len * seg_len)

            # Signed perpendicular distance via cross product
            cross_z = float(segment_vec[0] * to_obstacle[1] - segment_vec[1] * to_obstacle[0])
            signed_dist = cross_z / seg_len
            abs_dist = abs(signed_dist)

            # Prefer segments where the projection falls within bounds
            if 0.0 <= t <= 1.0:
                if abs_dist < best_bounded_abs:
                    best_bounded_abs = abs_dist
                    best_bounded_signed = signed_dist
            else:
                if abs_dist < best_unbounded_abs:
                    best_unbounded_abs = abs_dist
                    best_unbounded_signed = signed_dist

        # Prefer bounded match; fall back to unbounded if none found
        if best_bounded_abs < float('inf'):
            return best_bounded_signed
        return best_unbounded_signed

    def _intersect_rays_with_segments(self, origins, ray_ends, seg_starts, seg_ends):
        """
        Vectorized calculation of distances from origins to the first intersection
        of rays with a set of line segments.

        origins: (2,) array
        ray_ends: (N, 2) array (N = number of sectors)
        seg_starts: (M, 2) array (M = number of segments)
        seg_ends: (M, 2) array
        """
        # Ray vectors (N, 2)
        r = ray_ends - origins
        # Segment vectors (M, 2)
        s = seg_ends - seg_starts

        # Origins to segment starts (M, 2)
        origin_to_starts = seg_starts - origins

        # Cross product of r and s for all pairs (N, M)
        # Using broadcasting: r is (N, 1, 2) and s is (1, M, 2)
        denom = r[:, np.newaxis, 0] * s[np.newaxis, :, 1] - r[:, np.newaxis, 1] * s[np.newaxis, :, 0]

        # Avoid division by zero for parallel lines
        denom[np.abs(denom) < 1e-10] = np.nan

        # t = (q - p) x s / (r x s)
        t = (origin_to_starts[np.newaxis, :, 0] * s[np.newaxis, :, 1] -
             origin_to_starts[np.newaxis, :, 1] * s[np.newaxis, :, 0]) / denom

        # u = (q - p) x r / (r x s)
        u = (origin_to_starts[np.newaxis, :, 0] * r[:, np.newaxis, 1] -
             origin_to_starts[np.newaxis, :, 1] * r[:, np.newaxis, 0]) / denom

        # Valid intersection if 0 <= t <= 1 (on ray) and 0 <= u <= 1 (on segment)
        valid = (t >= 0) & (t <= 1) & (u >= 0) & (u <= 1)

        # Get distances for valid intersections, otherwise set to max_range
        dist_matrix = np.where(valid, t * self.polar_max_range, self.polar_max_range)

        # Return minimum distance for each ray across all segments
        return np.min(dist_matrix, axis=1)

    def _rebuild_polar_scan_geometries(self) -> None:
        """Prepare the boundary and collision-entity geometry for vectorized ray casting."""
        coords = np.array(self.obstacles, dtype=np.float32)

        if not np.allclose(coords[0], coords[-1]):
            coords = np.vstack([coords, coords[0]])

        self._boundary_starts = coords[:-1]
        self._boundary_ends = coords[1:]

        entities = getattr(self, "collision_entities", []) or []
        if entities:
            centers = np.array([entity.get_position() for entity in entities], dtype=np.float32)
            radii = np.array([float(entity.radius) for entity in entities], dtype=np.float32)
            self._collision_entity_centers = centers.reshape(-1, 2)
            self._collision_entity_radii_sq = np.square(radii + float(self.own_ship_radius))
        else:
            self._collision_entity_centers = None
            self._collision_entity_radii_sq = None

    def _get_polar_scan(self) -> np.ndarray:
        ship_pos = self.ship_pos
        x0, y0 = ship_pos
        heading = self.phys_sim.state.heading

        # Calculate ray endpoints
        ch, sh = np.cos(heading), np.sin(heading)
        dx = ch * self._polar_rel_dirs[:, 0] - sh * self._polar_rel_dirs[:, 1]
        dy = sh * self._polar_rel_dirs[:, 0] + ch * self._polar_rel_dirs[:, 1]

        ray_ends = np.stack([x0 + self.polar_max_range * dx,
                             y0 + self.polar_max_range * dy], axis=1)

        # Filter to relevant boundaries to reduce computation time
        buffer = 50.0
        limit_sq = (self.polar_max_range + buffer) ** 2
        d0 = (self._boundary_starts[:, 0] - x0) ** 2 + (self._boundary_starts[:, 1] - y0) ** 2
        d1 = (self._boundary_ends[:, 0] - x0) ** 2 + (self._boundary_ends[:, 1] - y0) ** 2

        mask = (d0 < limit_sq) | (d1 < limit_sq)
        local_starts = self._boundary_starts[mask]
        local_ends = self._boundary_ends[mask]

        # Initialize distances with max_range
        distances = np.full(self.polar_n_sectors, self.polar_max_range, dtype=np.float32)

        # Intersect with filtered boundaries
        if len(local_starts) > 0:
            distances = self._intersect_rays_with_segments(
                ship_pos, ray_ends, local_starts, local_ends
            )

        # Handle collision entities (circles)
        if self._collision_entity_centers is not None and len(self._collision_entity_centers) > 0:
            # Ray directions (N, 2)
            d = ray_ends - ship_pos

            # Ship to centers: (M, 2)
            f = ship_pos - self._collision_entity_centers

            # Quadratic components
            a = np.sum(d ** 2, axis=1)[:, np.newaxis]
            b = 2 * (d @ f.T)
            c = (np.sum(f ** 2, axis=1) - self._collision_entity_radii_sq)[np.newaxis, :]

            discriminant = b ** 2 - 4 * a * c
            hits = (discriminant >= 0)

            # Calculate t
            t = np.where(hits, (-b - np.sqrt(np.maximum(0, discriminant))) / (2 * a), 1.0)

            # Valid t is between 0 and 1
            valid_t = np.where((t >= 0) & (t <= 1), t * self.polar_max_range, self.polar_max_range)

            circle_distances = np.min(valid_t, axis=1)

            # Take the minimum of boundary hit vs circle hit
            distances = np.minimum(distances, circle_distances)

        dist = np.clip(distances / self.polar_max_range, 0.0, 1.0).astype(np.float32)
        closeness = 1.0 - dist
        scan = closeness * float(self.polar_obs_max)

        return np.clip(scan, 0.0, float(self.polar_obs_max)).astype(np.float32)

    # -----------------------
    # Observation & reset
    # -----------------------
    def _initialize_observation_space(self) -> gym.spaces.Box:
        """
        Extend the parent's observation space with polar scan sectors + CRI.
        """
        base_space = super()._initialize_observation_space()

        if not getattr(self, "use_polar_scan", False):
            return base_space

        # Append polar scan dimensions to the observation
        polar_low = np.zeros(self.polar_n_sectors, dtype=np.float32)
        polar_high = np.full(self.polar_n_sectors, self.polar_obs_max, dtype=np.float32)

        # Append CRI scalar (0 = no encounter / static obstacle, up to ~1.0 for dynamic TS)
        cri_low = np.zeros(1, dtype=np.float32)
        cri_high = np.ones(1, dtype=np.float32)

        extended_low = np.hstack([base_space.low, polar_low, cri_low])
        extended_high = np.hstack([base_space.high, polar_high, cri_high])

        # Optional one-hot scenario encoding for multi-policy agents
        if getattr(self, "include_scenario_one_hot", False):
            one_hot_low = np.zeros(NUM_SCENARIOS, dtype=np.float32)
            one_hot_high = np.ones(NUM_SCENARIOS, dtype=np.float32)
            extended_low = np.hstack([extended_low, one_hot_low])
            extended_high = np.hstack([extended_high, one_hot_high])

        return gym.spaces.Box(low=extended_low, high=extended_high, dtype=np.float32)

    def _get_obs(self) -> np.ndarray:
        """
        Extend the parent's observation with polar scan data and a CRI scalar.

        Returns:
            np.ndarray: shape (base_obs_dim + polar_n_sectors + 1,)
        """
        base_obs = super()._get_obs()

        if not getattr(self, "use_polar_scan", False):
            return base_obs

        # Get or compute polar scan
        scan = self.polar_scan_cached
        if scan is None:
            scan = self._get_polar_scan()
            self.polar_scan_cached = scan

        # Classify scenario (used by reward/violation logic, not in obs)
        if not self.phys_sim.state.terminated and not self.phys_sim.state.truncated:
            self.scenario_classifier.classify(
                self.ship_pos,
                self.phys_sim.state.heading,
                self.collision_entities,
            )

        # Compute CRI (0 when no tracked TS, else CRI of tracked TS)
        cri_value = 0.0
        clf = self.scenario_classifier
        if clf.tracked_ts is not None:
            ts = self.collision_entities[clf.tracked_ts]
            os_pos = self.ship_pos
            os_heading = self.phys_sim.state.heading
            os_u = float(self.phys_sim.state.u)
            os_v = float(self.phys_sim.state.v)
            cri_value = float(np.clip(self._compute_cri(os_pos, os_heading, os_u, os_v, ts), 0.0, 1.0))

        obs = np.hstack([base_obs, scan, np.array([cri_value], dtype=np.float32)]).astype(np.float32)

        # Append one-hot scenario encoding if enabled
        if getattr(self, "include_scenario_one_hot", False):
            obs = np.hstack([obs, self.scenario_classifier.get_one_hot()]).astype(np.float32)

        return obs

    # -----------------------
    # Dynamics update & step
    # -----------------------
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one environment timestep with collision entity
        updates inserted between physics and reward computation.
        """
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

        # Step collision entities (once, after the ship has moved)
        self._step_collision_entities()

        # Track minimum center-to-center distance to collision entities
        if getattr(self, "collision_entities", None) and getattr(self, "eval", False):
            for entity in self.collision_entities:
                dist = entity.distance_to_point(self.ship_pos)
                if dist < self._current_episode_min_dist:
                    self._current_episode_min_dist = dist

        # Update polar scan
        if self.use_polar_scan:
            if not self.phys_sim.state.terminated and not self.phys_sim.state.truncated:
                self.polar_scan_cached = self._get_polar_scan()

        # Compute live signed CTE for each static entity.
        # Stored as dict: _live_signed_cte = {entity_id: float}
        # Used by both reward and observation to avoid recomputation.
        self._live_signed_cte = {}
        if not self.phys_sim.state.terminated and not self.phys_sim.state.truncated:
            for entity in (self.collision_entities or []):
                if entity.is_dynamic:
                    continue
                obs_pos = np.array([entity.x, entity.y], dtype=np.float32)
                self._live_signed_cte[id(entity)] = self._nearest_segment_signed_cte(obs_pos)

        # Calculate reward and termination
        reward = 0.0
        terminated = False
        truncated = False

        if not self.phys_sim.state.terminated and not self.phys_sim.state.truncated:
            reward, terminated = self._calculate_reward_single()

            if self.step_count >= self.max_steps and not terminated:
                reward = -250
                truncated = True

            self.phys_sim.state.terminated = terminated
            self.phys_sim.state.truncated = truncated

        # Mark collision so the min distance for this episode is excluded from the PDF
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

    # -------------------
    # Reward calculation
    # -------------------
    def _calculate_reward_single(self) -> Tuple[float, bool]:
        """
        Extend parent's path-following reward with colav-specific penalties.
        """

        # Capture the active path segment before parent may advance checkpoint_index
        prev_idx = max(0, self.checkpoint_index - 1)
        collision_segment_start = np.asarray(self.checkpoints[prev_idx]['pos'], dtype=np.float32)
        curr_idx = min(self.checkpoint_index, len(self.checkpoints) - 1)
        collision_segment_end = np.asarray(self.checkpoints[curr_idx]['pos'], dtype=np.float32)

        # Base path-following reward and termination
        base_reward, done = super()._calculate_reward()
        reward = base_reward

        # ------------------------------------------------------------------ #
        #  Polar distance penalty                                              #
        # ------------------------------------------------------------------ #
        _step_polar = 0.0
        if getattr(self, "use_polar_scan", False) and self.polar_scan_cached is not None:
            max_close = float(np.max(self.polar_scan_cached))
            min_dist_n = 1.0 - (max_close / max(float(self.polar_obs_max), 1e-6))

            d_ramp_start = 100
            ramp_max_penalty = 5.0
            min_dist = min_dist_n * self.polar_max_range

            if min_dist < d_ramp_start:
                collision_boundary = 0.0
                ramp_t = (min_dist - collision_boundary) / max(d_ramp_start - collision_boundary, EPSILON)
                ramp_t = float(np.clip(ramp_t, 0.0, 1.0))
                exp_t = (np.exp(ramp_t) - 1.0) / (np.e - 1.0)
                ramp_penalty = ramp_max_penalty * (1.0 - exp_t)
                _step_polar = self.reward_weights['polar_distance'] * ramp_penalty
                reward -= _step_polar

        clf = self.scenario_classifier
        ts_idx = clf.tracked_ts

        # Reset crossing-left speed capture when scenario changes
        if clf.label != "crossing-left":
            self._crossing_left_initial_speed = None

        # Reset gravitation point when no longer crossing-right
        if clf.label != "crossing-right":
            self._crossing_right_grav_point = None

        if ts_idx is not None:
            ts = self.collision_entities[ts_idx]
            os_pos = self.ship_pos
            os_heading = self.phys_sim.state.heading
            scenario = clf.label

            if scenario == "head-on":
                ts_pos = ts.get_position()
                dx = float(ts_pos[0] - os_pos[0])
                dy = float(ts_pos[1] - os_pos[1])
                rel_bearing = (
                                      math.atan2(dy, dx) - os_heading + math.pi
                              ) % (2.0 * math.pi) - math.pi

                dist = ts.distance_to_point(os_pos)
                proximity = max(0.0, 1.0 - dist / clf.classify_range)
                reward += self.reward_weights["headon_bearing"] * math.sin(rel_bearing) * proximity

                # Logging only — no reward effect
                side = ts.ts_side_from_os_perspective(os_pos, os_heading)
                if ts.passed_by_os(os_pos, os_heading) and self.ts_passed != ts_idx:
                    self.ts_passed = ts_idx
                    if side == "port":
                        self._colreg_correct += 1
                    elif side == "starboard":
                        self._colreg_violations += 1

                    # Record passing snapshot for rendering
                    self._passing_snapshots.append({
                        "os_pos": np.copy(os_pos),
                        "ts_pos": ts_pos.copy(),
                        "dist": dist,
                        "scenario": scenario,
                    })

            elif scenario == "overtaking":
                side = ts.ts_side_from_os_perspective(os_pos, os_heading)
                if ts.passed_by_os(os_pos, os_heading) and self.ts_passed != ts_idx:
                    self.ts_passed = ts_idx
                    if side == "port":
                        self._colreg_correct += 1
                    elif side == "starboard":
                        self._colreg_violations += 1

                    # Record passing snapshot for rendering
                    dist = ts.distance_to_point(os_pos)
                    self._passing_snapshots.append({
                        "os_pos": np.copy(os_pos),
                        "ts_pos": ts.get_position().copy(),
                        "dist": dist,
                        "scenario": scenario,
                    })

            elif scenario == "crossing-right":
                ts_pos = ts.get_position()
                dx = float(ts_pos[0] - os_pos[0])
                dy = float(ts_pos[1] - os_pos[1])
                rel_bearing = (
                                      math.atan2(dy, dx) - os_heading + math.pi
                              ) % (2.0 * math.pi) - math.pi

                os_u = float(self.phys_sim.state.u)
                os_v = float(self.phys_sim.state.v)
                cri = self._compute_cri(os_pos, os_heading, os_u, os_v, ts)

                # Attenuate polar penalty: the TS appears in the polar scan
                # and penalizes the correct starboard turn. Keep 50% of the CRI
                # and offset with 20% so that at least 20% of the polar
                # distance penalty is kept
                polar_crossing_keep = 0.1 + 0.9 * cri
                reward += (1.0 - polar_crossing_keep) * _step_polar

                if -math.pi / 2 <= rel_bearing <= math.pi / 2:
                    crossing_signal = math.sin(rel_bearing)
                elif rel_bearing < -math.pi / 2:
                    crossing_signal = (-0.005 * (math.degrees(rel_bearing) + 90) ** 2) + math.sin(math.radians(-90))
                else:
                    crossing_signal = (-0.005 * (math.degrees(rel_bearing) - 90) ** 2) + math.sin(math.radians(90))
                crossing_reward = crossing_signal * (1.0 - cri)
                _step_crossing = self.reward_weights["crossing_right"] * crossing_reward
                reward += _step_crossing

                # Gravitation point: reward heading toward TS position at classification
                _step_grav = 0.0
                if self._crossing_right_grav_point is None:
                    self._crossing_right_grav_point = ts.get_position().copy()
                grav = self._crossing_right_grav_point
                grav_dx = float(grav[0] - os_pos[0])
                grav_dy = float(grav[1] - os_pos[1])
                # only active while OS is still approaching the gravitation point
                grav_approach = math.cos(os_heading) * grav_dx + math.sin(os_heading) * grav_dy
                if grav_approach > 0.0:
                    grav_bearing = (math.atan2(grav_dy, grav_dx) - os_heading + math.pi) % (2.0 * math.pi) - math.pi
                    _step_grav = self.reward_weights["crossing_right_grav"] * math.cos(grav_bearing)
                    reward += _step_grav

                # Violation: OS crossed ahead of TS (TS on starboard at passing)
                ts_pos = ts.get_position()
                side = ts.ts_side_from_os_perspective(os_pos, os_heading)
                if ts.passed_by_os(os_pos, os_heading) and self.ts_passed != ts_idx:
                    self.ts_passed = ts_idx
                    if side == "port":
                        self._colreg_correct += 1
                    elif side == "starboard":
                        self._colreg_violations += 1

                    dist = ts.distance_to_point(os_pos)
                    self._passing_snapshots.append({
                        "os_pos": np.copy(os_pos),
                        "ts_pos": ts_pos.copy(),
                        "dist": dist,
                        "scenario": scenario,
                    })

        # ------------------------------------------------------------------ #
        #  Collision detection                                                 #
        # ------------------------------------------------------------------ #
        collision_entity_collision = False
        if getattr(self, "collision_entities", None):
            collision_entity_collision = self.scenario_builder.check_collision_entity_collision()
            if collision_entity_collision:
                done = True

        bank_collision = not self._ship_in_open_water(self.ship_pos)
        self._last_collision_info = {}

        if bank_collision:
            self._last_collision_info = {"collision": True, "collision_type": "bank"}

        if collision_entity_collision:
            segment_vec = collision_segment_end - collision_segment_start
            ship_vec = self.ship_pos - collision_segment_start
            cross_z = float(segment_vec[0] * ship_vec[1] - segment_vec[1] * ship_vec[0])
            collision_side = "port" if cross_z > 0.0 else "starboard"
            cross_track_error = self._distance_from_point_to_line(
                self.ship_pos, collision_segment_start, collision_segment_end
            )
            self._last_collision_info = {
                "collision": True,
                "collision_type": "ship",
                "collision_cte": abs(cross_track_error),
                "collision_side": collision_side,
            }

        # Distinguish failure from success — only failures get -250
        success = self.checkpoint_index >= len(self.checkpoints)
        if done and not success:
            reward = -250.0

        return reward, done

    # -----------------------
    # Rendering
    # -----------------------
    def _initialize_rendering(self) -> None:
        super()._initialize_rendering()

        # Ship plot handle
        self.ship_plot_colav, = self.ax.plot([], [], 'bo', markersize=10, label='OS')
        self.heading_line_colav, = self.ax.plot([], [], color='black', linewidth=2)

        # Polar beams
        self.polar_beams = []
        if self.use_polar_scan:
            for _ in range(self.polar_n_sectors):
                beam_line, = self.ax.plot(
                    [], [],
                    color='orange',
                    alpha=0.3,
                    linewidth=1,
                )
                self.polar_beams.append(beam_line)

        # OS trail line
        self.os_trail_history = []
        self.os_trail_line, = self.ax.plot(
            [], [],
            color='blue',
            alpha=0.5,
            linewidth=1.5,
            linestyle='-',
            label='OS trail',
        )

        # TS (collision entity) trail lines — created dynamically
        self.ts_trails = []
        self.ts_trail_history = []

        # Passing snapshot artists (afterimages at the moment OS passes TS)
        self._passing_snapshot_artists = []

    def _get_collision_entity_render_key(self):
        """Small hashable signature of current collision-entity layout for render updates."""
        entities = getattr(self, "collision_entities", []) or []
        if not entities:
            return ("none", 0)

        rounded = tuple(
            (float(round(entity.x, 2)), float(round(entity.y, 2)), float(round(entity.radius, 2)), entity.entity_type)
            for entity in entities
        )
        return (len(entities), rounded)

    def _clear_collision_entity_patches(self) -> None:
        """Remove previous collision-entity patches from the axes."""
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
        """Draw current episode's collision entities and store patch handles."""
        if not hasattr(self, "ax"):
            return

        self._clear_collision_entity_patches()

        entities = getattr(self, "collision_entities", []) or []
        for i, entity in enumerate(entities):
            sx, sy = float(entity.x), float(entity.y)
            radius = float(entity.radius)

            safety = patches.Circle(
                (sx, sy),
                radius,
                fill=False,
                edgecolor="gray",
                linewidth=2,
                alpha=0.8,
                label="Collision entity" if i == 0 else ""
            )
            self.ax.add_patch(safety)
            self._collision_entity_patches.append(safety)

    def _draw_fixed_landmarks(self) -> None:
        self.start_pos = np.asarray(self.initial_ship_pos, dtype=np.float32).copy()
        super()._draw_fixed_landmarks()

    def render(self) -> None:
        """Render the environment and visualize ship movement."""
        if self.render_mode != 'human':
            return

        # Initialize plotting objects once
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

        # If collision entity layout changed, redraw and refresh background
        new_key = self._get_collision_entity_render_key()
        if new_key != getattr(self, "_collision_entity_render_key", None):
            self._collision_entity_render_key = new_key
            self._draw_collision_entities()
            self.fig.canvas.draw()
            self.background = self.fig.canvas.copy_from_bbox(self.ax.bbox)

        # Restore background for blit optimization
        if hasattr(self, 'background'):
            _safe(lambda: canvas.restore_region(self.background))

        heading_line_length = 30.0

        # Record OS position into trail history
        if not self.phys_sim.state.terminated and not self.phys_sim.state.truncated:
            self.os_trail_history.append(self.ship_pos.copy())

        # Record TS positions and ensure we have enough trail line objects
        entities = getattr(self, "collision_entities", []) or []
        # Grow trail lists if new entities appeared (new episode)
        while len(self.ts_trail_history) < len(entities):
            self.ts_trail_history.append([])
            trail_line, = self.ax.plot(
                [], [],
                color='red',
                alpha=0.5,
                linewidth=1.5,
                linestyle='--',
                label='TS trail' if len(self.ts_trails) == 0 else '',
            )
            self.ts_trails.append(trail_line)

        for idx, entity in enumerate(entities):
            self.ts_trail_history[idx].append(entity.get_position().copy())

        # Update OS trail line
        if self.os_trail_history:
            pts = np.array(self.os_trail_history)
            self.os_trail_line.set_data(pts[:, 0], pts[:, 1])

        # Update TS trail lines
        for idx in range(len(entities)):
            if self.ts_trail_history[idx]:
                pts = np.array(self.ts_trail_history[idx])
                self.ts_trails[idx].set_data(pts[:, 0], pts[:, 1])

        # Update ship marker
        self.ship_plot_colav.set_data(self.ship_pos[0:1], self.ship_pos[1:2])

        # Update heading line
        heading = self.phys_sim.state.heading
        hx = self.ship_pos[0] + np.cos(heading) * heading_line_length
        hy = self.ship_pos[1] + np.sin(heading) * heading_line_length
        self.heading_line_colav.set_data(
            [self.ship_pos[0], hx],
            [self.ship_pos[1], hy]
        )

        # Update polar beams
        if self.use_polar_scan and self.polar_beams:
            scan = self.polar_scan_cached
            if scan is None:
                scan = self._get_polar_scan()

            sx = float(self.ship_pos[0])
            sy = float(self.ship_pos[1])

            for j, beam in enumerate(self.polar_beams):
                rel_angle = float(self._polar_rel_angles[j])
                angle = heading + rel_angle
                closeness01 = float(scan[j]) / max(float(self.polar_obs_max), 1e-6)
                length = (1.0 - closeness01) * float(self.polar_max_range)
                beam.set_data(
                    [sx, sx + length * np.cos(angle)],
                    [sy, sy + length * np.sin(angle)]
                )

        # Create artists for any new passing snapshots
        while len(self._passing_snapshot_artists) < len(self._passing_snapshots):
            idx = len(self._passing_snapshot_artists)
            snap = self._passing_snapshots[idx]
            os_p = snap["os_pos"]
            ts_p = snap["ts_pos"]
            dist = snap["dist"]

            # Afterimage markers (hollow, larger)
            os_marker, = self.ax.plot(
                [os_p[0]], [os_p[1]], 'o',
                markersize=12, markerfacecolor='none', markeredgecolor='blue',
                markeredgewidth=2, zorder=5,
            )
            ts_marker, = self.ax.plot(
                [ts_p[0]], [ts_p[1]], 'o',
                markersize=12, markerfacecolor='none', markeredgecolor='red',
                markeredgewidth=2, zorder=5,
            )
            # Dashed line between the two
            dist_line, = self.ax.plot(
                [os_p[0], ts_p[0]], [os_p[1], ts_p[1]],
                color='gray', linestyle='--', linewidth=1.2, zorder=4,
            )
            # Distance label at midpoint
            mid_x = (os_p[0] + ts_p[0]) / 2.0
            mid_y = (os_p[1] + ts_p[1]) / 2.0
            dist_text = self.ax.text(
                mid_x, mid_y, f"{dist:.0f}m",
                fontsize=7, color='gray', fontweight='bold',
                ha='center', va='bottom', zorder=6,
            )
            # OS / TS labels
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

        # Blit all artists
        try:
            if hasattr(self.fig.canvas, 'blit'):
                # Draw trail first (underneath markers)
                self.ax.draw_artist(self.os_trail_line)
                for tl in self.ts_trails:
                    self.ax.draw_artist(tl)

                # Draw passing snapshot afterimages
                for artists in self._passing_snapshot_artists:
                    for artist in artists:
                        self.ax.draw_artist(artist)

                # Draw ship, heading, and beams on top
                self.ax.draw_artist(self.ship_plot_colav)
                self.ax.draw_artist(self.heading_line_colav)
                if self.use_polar_scan:
                    for beam in self.polar_beams:
                        self.ax.draw_artist(beam)
                self.fig.canvas.blit(self.ax.bbox)
            else:
                self.fig.canvas.draw()
        except (RuntimeError, AttributeError, TypeError, ValueError):
            _safe(self.fig.canvas.draw)
        _safe(canvas.flush_events)

    # -----------------------
    # Collision-entity helpers
    # -----------------------
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

        # Active ship position for distance checks
        active_ship_positions = []
        if not self.phys_sim.state.terminated and not self.phys_sim.state.truncated:
            active_ship_positions = [self.ship_pos]

        for entity in self.collision_entities:
            if not getattr(entity, "is_dynamic", False):
                continue

            old_pos = entity.get_position().copy()

            if getattr(entity, "entity_type", None) == "head-on": #head-on
                # React to the nearest active OS that has not been dodged yet
                 if active_ship_positions and not getattr(entity, "dodged", False):
                     entity_pos = entity.get_position()
                     min_dist = min(
                         float(np.linalg.norm(entity_pos - os_pos))
                         for os_pos in active_ship_positions
                     )
                     # if TS gets into half of the polar max range, turn starboard and do not follow
                     # the route anymore
                     if min_dist <= self.polar_max_range / 2:
                         entity.dodged = True
                         dodge_angle = math.radians(20)  # 20° starboard
                         entity.set_heading(entity.get_heading() - dodge_angle)
                         entity.set_lateral_speed(0.0)
                         entity.route_mode = None

            # Activate dormant crossing entities when OS gets within activation
            # range AND the crossing point is still ahead of the OS.
            # At activation, compute the collision-course speed dynamically from
            # the actual OS-to-crossing-point distance so path curvature doesn't
            # break the intercept timing.
            if getattr(entity, "entity_type", "") in ("crossing-right", "crossing-left"):
                if not getattr(entity, "crossing_activated", False) and active_ship_positions:
                    entity_pos = entity.get_position()
                    crossing_point = getattr(entity, "crossing_point", None)

                    # Find the nearest OS and its distance to the TS
                    best_os_pos = None
                    best_dist = float("inf")
                    for os_pos in active_ship_positions:
                        d = float(np.linalg.norm(entity_pos - os_pos))
                        if d < best_dist:
                            best_dist = d
                            best_os_pos = os_pos

                    activation_range = getattr(entity, "crossing_activation_range", self.polar_max_range + 50.0)

                    # For narrow encounters (small d_ts), cap the activation
                    # distance so the speed formula doesn't get clamped by
                    # crossing_ts_v_min.  d_os_max = sf * d_ts * v_os / v_min
                    # ensures v_ts >= v_min without clamping, preserving the
                    # intended arrival-time ratio.
                    d_ts = float(getattr(entity, "crossing_d_ts", 0))
                    if d_ts > 0 and self.crossing_ts_v_min > 0:
                        sf = float(getattr(self, "crossing_speed_factor", 1.0))
                        v_os_est = float(max(abs(self.phys_sim.state.u), self.os_nominal_speed * 0.5))
                        timing_cap = sf * d_ts * v_os_est / self.crossing_ts_v_min
                        activation_range = min(activation_range, timing_cap)

                    if best_os_pos is not None and best_dist <= activation_range and crossing_point is not None:
                        # Check that the crossing point is still ahead of the OS
                        # along the path direction (not OS heading, which may diverge
                        # on curved paths near canal banks).
                        to_cp = crossing_point - best_os_pos
                        path_tangent = getattr(entity, "crossing_path_tangent", None)
                        if path_tangent is not None:
                            forward = np.asarray(path_tangent, dtype=np.float32)
                        else:
                            os_heading = float(self.phys_sim.state.heading)
                            forward = np.array([math.cos(os_heading), math.sin(os_heading)], dtype=np.float32)
                        ahead_dot = float(np.dot(to_cp, forward))

                        if ahead_dot > 0.0:
                            # Compute collision-course speed using actual OS speed,
                            # ensures the TS is always on a collision
                            # course regardless of what speed the agent has learned —
                            # slowing down won't make the threat disappear.
                            # v_ts = speed_factor * d_ts * v_os_actual / d_os
                            d_os = float(np.linalg.norm(to_cp))
                            v_os_actual = float(max(abs(self.phys_sim.state.u), self.os_nominal_speed * 0.5))
                            speed_factor = float(getattr(self, "crossing_speed_factor", 1.0))
                            if d_os > EPSILON:
                                v_ts = speed_factor * d_ts * v_os_actual / d_os
                            else:
                                v_ts = float(self.crossing_ts_v_max)

                            # Clamp to configured speed bounds
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

            if getattr(self, "use_polar_scan", False):
                self._rebuild_polar_scan_geometries()
                self.polar_scan_cached = None

    def _advance_entity_along_checkpoints(self, entity, dt: float) -> None:
        remaining_dist = float(entity.path_speed) * float(dt)
        path_direction = int(getattr(entity, "path_direction", 1))
        path_offset = float(getattr(entity, "path_offset", 0.0))

        while remaining_dist > EPSILON and not getattr(entity, "route_finished", False):
            next_idx = int(getattr(entity, "next_checkpoint_index", -1))

            # valid range depends on travel direction
            if next_idx < 0 or next_idx >= len(self.checkpoints):
                entity.route_finished = True
                entity.set_speed(0.0)
                return

            # Centerline travel direction for the current segment
            # Used in the takeover scenario
            if path_direction >= 0:
                # moving forward toward checkpoint[next_idx]
                if next_idx <= 0:
                    entity.route_finished = True
                    entity.set_speed(0.0)
                    return
                cp_a = np.asarray(self.checkpoints[next_idx - 1]["pos"], dtype=np.float32)
                cp_b = np.asarray(self.checkpoints[next_idx]["pos"], dtype=np.float32)
                travel_dir = cp_b - cp_a
            else:
                # moving backward toward checkpoint[next_idx]
                # used in the head-on scenario
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

            # Starboard relative to the entity heading/travel direction
            starboard = np.array([travel_dir[1], -travel_dir[0]], dtype=np.float32)

            # Offset target point from the path centerline
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

    def env_specific_reset(self) -> None:
        """Reset episode-specific geometry and build collision entities for the episode."""
        # Finalize previous episode's min distance (skip the very first reset and collisions)
        if hasattr(self, "_current_episode_min_dist") and getattr(self, "eval", False):
            if not self._current_episode_collided and self._current_episode_min_dist < float("inf"):
                self._episode_min_distances.append(self._current_episode_min_dist)
        self._current_episode_min_dist = float("inf")
        self._current_episode_collided = False

        if getattr(self, "use_polar_scan", False):
            self.polar_scan_cached = None

        self.collision_entities = []
        self.obstacle_count = 0
        self.obstacle_pos = None
        self.static_ship_radii = []

        # Reset collision info
        self._last_collision_info = {}

        # Reset scenario classifier and COLREG reward tracking
        self.scenario_classifier.reset()
        self.ts_passed = None
        self._colreg_violations = 0
        self._colreg_correct = 0
        self._colreg_initial_heading = None
        self._crossing_left_initial_speed = None
        self._crossing_right_grav_point = None

        # Remove passing snapshot artists from the plot and reset
        for artists in getattr(self, "_passing_snapshot_artists", []):
            for artist in artists:
                artist.remove()
        self._passing_snapshot_artists = []
        self._passing_snapshots = []

        if getattr(self, "enable_obstacles", False):
            if getattr(self, "eval", False):
                self.obstacle_count = int(self.eval_obstacle_count)
                self.collision_entities = self.scenario_builder.compute_collision_entity_positions(
                    self.eval_scenario, self.obstacle_count
                ) or []
                self._sync_collision_entity_views()
            else:
                self.scenario_builder.sample_collision_entity_scenario()

        if getattr(self, "use_polar_scan", False):
            self._rebuild_polar_scan_geometries()

        self._overtaking_pass_reported = False
        self._os_speed_samples = []

        # Clear trail histories for the new episode
        self.os_trail_history = []
        for trail_line in getattr(self, "ts_trails", []):
            trail_line.remove()
        self.ts_trail_history = []
        self.ts_trails = []

    def plot_min_distance_PDF(self, save_path: str = "min_distance_PDF.png") -> None:
        """
        Plot a probability density estimate of the minimum center-to-center
        distance kept to collision entities across non-collision evaluation
        episodes, with arrows on the x-axis indicating the collision boundaries.

        Call ``finalize_eval_min_distances()`` first to include the last episode.
        """
        if not self._current_episode_collided and self._current_episode_min_dist < float("inf"):
            self._episode_min_distances.append(self._current_episode_min_dist)

        distances = np.array(self._episode_min_distances)
        if len(distances) == 0:
            print("No non-collision episodes to plot min-distance png for.")
            return

        fig, ax = plt.subplots(figsize=(10, 6))

        # Histogram as density (area sums to 1)
        n_bins = max(10, len(distances) // 3)
        ax.hist(distances, bins=n_bins, density=True, alpha=0.6, color="steelblue",
                edgecolor="white", label="Histogram (density)")

        # KDE overlay for a smooth estimate
        kde = gaussian_kde(distances)
        x_grid = np.linspace(max(0, distances.min() - 5), distances.max() + 10, 300)
        ax.plot(x_grid, kde(x_grid), color="darkblue", linewidth=2, label="KDE")

        arrow_coords = ("data", "axes fraction")
        arrow_kwargs = dict(arrowstyle="<->", lw=1.5)
        text_kwargs = dict(ha="center", va="top", fontsize=8, fontweight="bold")

        os_r = float(self.own_ship_radius)
        ts_r_min = float(self.static_ship_radius_min)
        ts_r_max = float(self.static_ship_radius_max)
        variable_ts = ts_r_min != ts_r_max

        # Row 1: OS radius arrow from 0 to own_ship_radius
        row1_y = -0.06
        if os_r > 0:
            ax.annotate("", xy=(os_r, row1_y), xytext=(0, row1_y),
                         arrowprops=dict(**arrow_kwargs, color="red"),
                         xycoords=arrow_coords, textcoords=arrow_coords, clip_on=False)
            ax.text(os_r / 2, row1_y - 0.02, f"OS radius = {os_r:.0f}m",
                    transform=ax.get_xaxis_transform(), color="red",
                    clip_on=False, **text_kwargs)

        # Row 2: TS radius arrow(s) starting at OS radius
        row2_y = row1_y - 0.08
        if variable_ts:
            # Min TS radius arrow
            ax.annotate("", xy=(os_r + ts_r_min, row2_y), xytext=(os_r, row2_y),
                         arrowprops=dict(**arrow_kwargs, color="orange"),
                         xycoords=arrow_coords, textcoords=arrow_coords, clip_on=False)
            ax.text(os_r + ts_r_min / 2, row2_y - 0.02,
                    f"min TS radius = {ts_r_min:.0f}m",
                    transform=ax.get_xaxis_transform(), color="orange",
                    clip_on=False, **text_kwargs)
            # Max TS radius arrow (one more row down)
            row3_y = row2_y - 0.08
            ax.annotate("", xy=(os_r + ts_r_max, row3_y), xytext=(os_r, row3_y),
                         arrowprops=dict(**arrow_kwargs, color="darkorange"),
                         xycoords=arrow_coords, textcoords=arrow_coords, clip_on=False)
            ax.text(os_r + ts_r_max / 2, row3_y - 0.02,
                    f"max TS radius = {ts_r_max:.0f}m",
                    transform=ax.get_xaxis_transform(), color="darkorange",
                    clip_on=False, **text_kwargs)
        else:
            ax.annotate("", xy=(os_r + ts_r_min, row2_y), xytext=(os_r, row2_y),
                         arrowprops=dict(**arrow_kwargs, color="orange"),
                         xycoords=arrow_coords, textcoords=arrow_coords, clip_on=False)
            ax.text(os_r + ts_r_min / 2, row2_y - 0.02,
                    f"TS radius = {ts_r_min:.0f}m",
                    transform=ax.get_xaxis_transform(), color="orange",
                    clip_on=False, **text_kwargs)

        # Vertical dashed lines at collision boundaries
        collision_dist_min = os_r + ts_r_min
        collision_dist_max = os_r + ts_r_max
        ax.axvline(collision_dist_min, color="red", linestyle="--", alpha=0.6,
                   label=f"Collision boundary (min) = {collision_dist_min:.0f}m")
        if variable_ts:
            ax.axvline(collision_dist_max, color="darkorange", linestyle="--", alpha=0.6,
                       label=f"Collision boundary (max) = {collision_dist_max:.0f}m")

        ax.set_ylim(bottom=0)
        ax.set_xlim(left=0)

        ax.set_xlabel("Minimum center-to-center distance to collision entity [m]")
        ax.set_ylabel("Probability density")
        ax.set_title(f"Min-distance PDF ({len(distances)} non-collision episodes)")
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Min-distance PDF saved as '{save_path}'")
        plt.close(fig)
