import matplotlib
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import gymnasium as gym
from fw.collision_avoidance.scenario_builder import ScenarioBuilder

import math

from fw.environments.py_sim_env import PySimEnv

# First try interactive backend
try:
    matplotlib.use('TkAgg')  # Try using the interactive backend
except ImportError:
    print("Warning: 'TkAgg' backend not available. Falling back to 'Agg'.")
    matplotlib.use('Agg')  # Use non-interactive backend as fallback

from functools import lru_cache
from shapely.geometry import Point, Polygon
from typing import List, Optional, Union
from fw.environments.tools import create_checkpoints_from_simple_path
from fw.ships.ship_factory import create_ship, ShipModel
from fw.simulators.ship_simulator import ShipSimulator
from fw.ships.vessel_state import VesselState
from fw.environments.base_env import BaseEnv
from fw.dynamics.dynamics_model import Model
from fw.dynamics.fossen import Fossen
from fw.dynamics.nomoto import Nomoto
import os
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
            eval_scenario: Optional[str] = "overtaking", # "static", "overtaking", "head-on"
            eval_obstacle_count: int = 5,
            num_ships: int = 1
    ) -> None:
        self.num_ships = int(num_ships)
        # Set colav attrs needed by overridden methods that super().__init__ may call
        self.enable_obstacles = bool(enable_obstacles)
        self.collision_entities = []
        self.obstacle_count = 0
        self.obstacle_pos = None
        self.obstacle_population = [0, 1, 2, 3]
        self.obstacle_weights = [50, 25, 15, 10]
        self.obstacle_probs = np.asarray(self.obstacle_weights, dtype=np.float32)
        self.obstacle_probs /= self.obstacle_probs.sum()
        self.static_ship_lateral_max = 0.0

        self.enable_variable_ship_size = bool(enable_variable_ship_size)
        self.static_ship_radius = 20
        self.static_ship_radius_min = 20.0
        self.static_ship_radius_max = 50.0 if self.enable_variable_ship_size else self.static_ship_radius_min
        self.static_ship_radii = []
        self.static_ship_segment_margin_frac = 0.1

        self.os_nominal_speed = 2.0
        self.overtaking_ts_v_min = 1.0
        self.overtaking_ts_v_max = self.os_nominal_speed - 0.3
        self.headon_ts_v_min = self.os_nominal_speed - 1.0
        self.headon_ts_v_max = self.os_nominal_speed + 1.0
        self.own_ship_radius = 0.0

        self.use_polar_scan = bool(use_polar_scan)
        self.polar_scan_cached = None
        self.polar_n_sectors = 54
        self.polar_fov = np.radians(270)
        self.polar_max_range = 250.0
        self.polar_obs_max = 0.1

        self.eval_obstacle_count = int(eval_obstacle_count)
        self.eval_scenario = str(eval_scenario)

        self._last_collision_info = [{} for _ in range(self.num_ships)]


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
            num_ships=num_ships,
        )

        # single-ship aliases expected by the existing colav code
        self.os_i = 0
        self.polar_scan_cached = [None] * self.num_ships

        self.scenario_builder = ScenarioBuilder(self)
        self.scenario_builder.cache_checkpoint_segments()

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
            "forward": 1.0,
            "alignment": 1.8,
            "cross_track": 1.5,
            "rudder": 0.03,
            "polar_distance": 3.0,
        }

        self._collision_entity_patches = []
        self._collision_entity_render_key = None

    # -------------------------
    # Utility & Math
    # -------------------------
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

    def _get_polar_scan(self, ship_index: int) -> np.ndarray:
        ship_pos = self.ship_pos[ship_index]
        x0, y0 = ship_pos
        heading = self.phys_sims[ship_index].state.heading

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
        Extend the parent's observation space with polar scan sectors.
        """
        base_space = super()._initialize_observation_space()

        if not getattr(self, "use_polar_scan", False):
            return base_space

        # Append polar scan dimensions to each ship's observation
        # base_space.low/high shape: (num_ships, base_obs_dim)
        polar_low = np.zeros((self.num_ships, self.polar_n_sectors), dtype=np.float32)
        polar_high = np.full((self.num_ships, self.polar_n_sectors), self.polar_obs_max, dtype=np.float32)

        extended_low = np.hstack([base_space.low, polar_low])
        extended_high = np.hstack([base_space.high, polar_high])

        return gym.spaces.Box(low=extended_low, high=extended_high, dtype=np.float32)

    def _get_obs(self) -> np.ndarray:
        """
        Extend the parent's multi-agent observation with per-ship polar scan data.

        Returns:
            np.ndarray: shape (num_ships, base_obs_dim + polar_n_sectors)
        """
        # Parent builds the full path-following observation for every ship
        # Shape: (num_ships, base_obs_dim)
        base_obs = super()._get_obs()

        if not getattr(self, "use_polar_scan", False):
            return base_obs

        # Append each ship's polar scan to its observation row
        scans = []
        for i in range(self.num_ships):
            scan = self.polar_scan_cached[i]
            if scan is None:
                scan = self._get_polar_scan(i)
                self.polar_scan_cached[i] = scan
            scans.append(scan)

        polar_block = np.array(scans, dtype=np.float32)  # (num_ships, polar_n_sectors)
        obs = np.hstack([base_obs, polar_block]).astype(np.float32)
        return obs

    # -----------------------
    # Dynamics update & step
    # -----------------------
    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one environment timestep for all ships, with collision entity
        updates inserted between physics and reward computation.
        """
        actions = np.asarray(actions, dtype=np.float32)
        if actions.shape == (2,):
            actions = actions[None, :]  # single-ship fallback

        rewards = np.zeros(self.num_ships, dtype=np.float32)
        terminations = np.zeros(self.num_ships, dtype=bool)
        truncations = np.zeros(self.num_ships, dtype=bool)

        # Step physics for all active ships
        for i in range(self.num_ships):
            if not self.phys_sims[i].state.terminated and not self.phys_sims[i].state.truncated:
                self.previous_action[i] = self.performed_action[i].copy()
                self.previous_ship_pos[i] = self.ship_pos[i].copy()
                self.previous_heading[i] = self.phys_sims[i].state.heading

                self.phys_sims[i].step(actions[i], self.enable_smoothing)
                self.ship_pos[i] = np.array(
                    [self.phys_sims[i].state.x, self.phys_sims[i].state.y], dtype=np.float32
                )
                self.performed_action[i] = self.phys_sims[i].actuators.current_actuator_state.copy()
                self.step_count[i] += 1

        # Step collision entities (once, after all ships have moved)
        self._step_collision_entities()

        # Update polar scans for all active ships
        if self.use_polar_scan:
            for i in range(self.num_ships):
                if not self.phys_sims[i].state.terminated and not self.phys_sims[i].state.truncated:
                    self.polar_scan_cached[i] = self._get_polar_scan(i)

        # Calculate rewards and termination for all active ships
        for i in range(self.num_ships):
            if not self.phys_sims[i].state.terminated and not self.phys_sims[i].state.truncated:
                reward, terminated = self._calculate_reward_single(i)

                truncated = False
                if self.step_count[i] >= self.max_steps and not terminated:
                    reward = -250
                    truncated = True

                self.phys_sims[i].state.terminated = terminated
                self.phys_sims[i].state.truncated = truncated

                rewards[i] = reward
                terminations[i] = terminated
                truncations[i] = truncated
            else:
                rewards[i] = 0.0
                terminations[i] = self.phys_sims[i].state.terminated
                truncations[i] = self.phys_sims[i].state.truncated

        info = {
            "rewards": rewards,
            "terminations": terminations,
            "truncations": truncations,
            "collision_info": [self._last_collision_info[i] for i in range(self.num_ships)],
        }

        return self._get_obs(), rewards[0], np.all(terminations | truncations), False, info

    # -------------------
    # Reward calculation
    # -------------------
    def _calculate_reward_single(self, i: int) -> Tuple[float, bool]:
        """
        Extend parent's path-following reward with colav-specific penalties.
        """
        # Capture the active path segment before parent may advance checkpoint_index
        prev_idx = max(0, self.checkpoint_index[i] - 1)
        collision_segment_start = np.asarray(self.checkpoints[prev_idx]['pos'], dtype=np.float32)
        curr_idx = min(self.checkpoint_index[i], len(self.checkpoints) - 1)
        collision_segment_end = np.asarray(self.checkpoints[curr_idx]['pos'], dtype=np.float32)

        # Base path-following reward and termination
        base_reward, done = super()._calculate_reward_single(i)
        reward = base_reward

        # Colav extension: polar distance penalty
        if getattr(self, "use_polar_scan", False) and self.polar_scan_cached[i] is not None:
            max_close = float(np.max(self.polar_scan_cached[i]))
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
                reward -= self.reward_weights['polar_distance'] * ramp_penalty

        # Colav extension: collision entity collision
        collision_entity_collision = False
        if getattr(self, "collision_entities", None):
            collision_entity_collision = self.scenario_builder.check_collision_entity_collision(ship_index=i)
            if collision_entity_collision:
                done = True

        # Build collision info for step() to include in its info dict
        bank_collision = not self._ship_in_open_water(self.ship_pos[i])
        self._last_collision_info[i] = {}

        if bank_collision:
            self._last_collision_info[i] = {"collision": True, "collision_type": "bank"}

        if collision_entity_collision:
            segment_vec = collision_segment_end - collision_segment_start
            ship_vec = self.ship_pos[i] - collision_segment_start
            cross_z = float(segment_vec[0] * ship_vec[1] - segment_vec[1] * ship_vec[0])
            collision_side = "port" if cross_z > 0.0 else "starboard"
            cross_track_error = self._distance_from_point_to_line(
                self.ship_pos[i], collision_segment_start, collision_segment_end
            )
            self._last_collision_info[i] = {
                "collision": True,
                "collision_type": "ship",
                "collision_cte": abs(cross_track_error),
                "collision_side": collision_side,
            }

        # Distinguish failure from success — only failures get -250
        success = self.checkpoint_index[i] >= len(self.checkpoints)
        if done and not success:
            reward = -250.0

        return reward, done

    # -----------------------
    # Rendering
    # -----------------------
    def _initialize_rendering(self) -> None:
        super()._initialize_rendering()

        # Per-ship plot handles (parent only creates one)
        colors = ['bo', 'co', 'mo', 'yo', 'ko']
        self.ship_plots = []
        self.heading_lines = []
        for i in range(self.num_ships):
            color = colors[i % len(colors)]
            plot, = self.ax.plot([], [], color, markersize=10, label=f'Ship {i}')
            heading, = self.ax.plot([], [], color='black', linewidth=2)
            self.ship_plots.append(plot)
            self.heading_lines.append(heading)

        # Per-ship polar beams
        self.polar_beams_per_ship = []
        if self.use_polar_scan:
            beam_colors = ['orange', 'cyan', 'magenta', 'yellow']
            for i in range(self.num_ships):
                beams = []
                beam_color = beam_colors[i % len(beam_colors)]
                for _ in range(self.polar_n_sectors):
                    beam_line, = self.ax.plot(
                        [], [],
                        color=beam_color,
                        alpha=0.3,
                        linewidth=1,
                    )
                    beams.append(beam_line)
                self.polar_beams_per_ship.append(beams)

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
        self.start_pos = np.asarray(self.initial_ship_pos[0], dtype=np.float32).copy()
        super()._draw_fixed_landmarks()

    def render(self) -> None:
        """Render the environment and visualize all ships' movement."""
        if self.render_mode != 'human':
            return

        # Initialize plotting objects once
        if self.initialize_plots and not hasattr(self, 'ship_plots'):
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

        for i in range(self.num_ships):
            # Update ship marker
            self.ship_plots[i].set_data(self.ship_pos[i][0:1], self.ship_pos[i][1:2])

            # Update heading line
            heading = self.phys_sims[i].state.heading
            hx = self.ship_pos[i][0] + np.cos(heading) * heading_line_length
            hy = self.ship_pos[i][1] + np.sin(heading) * heading_line_length
            self.heading_lines[i].set_data(
                [self.ship_pos[i][0], hx],
                [self.ship_pos[i][1], hy]
            )

            # Update polar beams for this ship
            if self.use_polar_scan and i < len(self.polar_beams_per_ship):
                scan = self.polar_scan_cached[i]
                if scan is None:
                    scan = self._get_polar_scan(i)

                sx = float(self.ship_pos[i][0])
                sy = float(self.ship_pos[i][1])

                for j, beam in enumerate(self.polar_beams_per_ship[i]):
                    rel_angle = float(self._polar_rel_angles[j])
                    angle = heading + rel_angle
                    closeness01 = float(scan[j]) / max(float(self.polar_obs_max), 1e-6)
                    length = (1.0 - closeness01) * float(self.polar_max_range)
                    beam.set_data(
                        [sx, sx + length * np.cos(angle)],
                        [sy, sy + length * np.sin(angle)]
                    )

        # Blit all artists
        try:
            if hasattr(self.fig.canvas, 'blit'):
                for i in range(self.num_ships):
                    self.ax.draw_artist(self.ship_plots[i])
                    self.ax.draw_artist(self.heading_lines[i])
                    if self.use_polar_scan and i < len(self.polar_beams_per_ship):
                        for beam in self.polar_beams_per_ship[i]:
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

        # Precompute active ship positions for distance checks
        active_ship_positions = [
            self.ship_pos[i]
            for i in range(self.num_ships)
            if not self.phys_sims[i].state.terminated and not self.phys_sims[i].state.truncated
        ]

        for entity in self.collision_entities:
            if not getattr(entity, "is_dynamic", False):
                continue

            old_pos = entity.get_position().copy()

            if getattr(entity, "entity_type", None) == "head-on":
                # React to the nearest active OS
                if active_ship_positions:
                    entity_pos = entity.get_position()
                    min_dist = min(
                        float(np.linalg.norm(entity_pos - os_pos))
                        for os_pos in active_ship_positions
                    )
                    if min_dist <= self.polar_max_range:
                        entity.set_lateral_speed(1)
                        entity.route_mode = None

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
                self.polar_scan_cached = [None] * self.num_ships

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
        if getattr(self, "use_polar_scan", False):
            self.polar_scan_cached = [None] * self.num_ships  # was: None

        self.collision_entities = []
        self.obstacle_count = 0
        self.obstacle_pos = None
        self.static_ship_radii = []

        # Reset per-ship collision info (from point 5)
        self._last_collision_info = [{} for _ in range(self.num_ships)]

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
