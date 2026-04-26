"""Polar rangefinder collision avoidance environment.

Extends PySimColavBaseEnv with a polar rangefinder observation and
COLREG-aware reward shaping for the polar/rangefinder approach.
"""
import matplotlib
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import gymnasium as gym
from fw.collision_avoidance.scenario_classifier import NUM_SCENARIOS, EXTENDED_SCENARIO_LABELS
from scipy.stats import gaussian_kde

import math

from fw.environments.py_sim_colav_base import PySimColavBaseEnv

try:
    matplotlib.use('TkAgg')
except ImportError:
    print("Warning: 'TkAgg' backend not available. Falling back to 'Agg'.")
    matplotlib.use('Agg')

from typing import Optional, Tuple, Dict
from fw.ships.ship_factory import ShipModel
from fw.dynamics.dynamics_model import Model
import numpy as np

EPSILON = 1e-9


class PySimColavPolarEnv(PySimColavBaseEnv):
    """Collision avoidance environment using polar rangefinder observations.

    Adds a polar scan (ray-cast sectors) and CRI scalar to the observation,
    and applies COLREG-aware reward shaping based on relative bearings.
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
            enable_obstacles: bool = True,
            use_polar_scan: bool = True,
            enable_variable_ship_size: bool = False,
            eval: bool = True,
            include_scenario_one_hot: bool = False,
            # Scenario configuration (passed to base)
            scenario_types: Optional[list] = None,
            scenario_probs: Optional[list] = None,
            obstacle_population: Optional[list] = None,
            obstacle_weights: Optional[list] = None,
            scenario_obstacle_level: Optional[str] = None,
            ground_truth_classification: bool = True,
            **kwargs,
    ) -> None:

        # Polar scan configuration
        self.use_polar_scan = bool(use_polar_scan)
        self.polar_scan_cached = None
        self.polar_n_sectors = 54
        self.polar_fov = np.radians(270)
        self.polar_max_range = 500.0
        self.polar_obs_max = 0.1 * self.polar_max_range / 250

        self.include_scenario_one_hot = bool(include_scenario_one_hot)

        # Polar-specific reward state
        self._colreg_initial_heading = None
        self._crossing_left_initial_speed = None
        self._crossing_right_grav_point = None

        # Pre-compute polar ray directions
        bin_width = float(self.polar_fov) / float(self.polar_n_sectors)
        self._polar_rel_angles = (
                -(self.polar_fov / 2.0) + (np.arange(self.polar_n_sectors, dtype=np.float32) + 0.5) * bin_width
        ).astype(np.float32)
        self._polar_rel_dirs = np.stack(
            (np.cos(self._polar_rel_angles), np.sin(self._polar_rel_angles)), axis=1
        ).astype(np.float32)

        # Scenario parameters
        scenario_obstacle_level = str(scenario_obstacle_level) if scenario_obstacle_level is not None else "step"
        scenario_probs = scenario_probs if scenario_probs is not None else [
            0.0, 0.0, 0.0, 0.6, 0.4, 0.0 # "path-following", "static", "overtaking", "head-on", "crossing-right", "crossing-left"
        ]

        # Call base init (encounter_range = polar_max_range)
        super().__init__(
            render_mode=render_mode,
            time_step=time_step,
            max_steps=max_steps,
            verbose=verbose,
            ship_pos=ship_pos,
            target_pos=target_pos,
            dynamics_model=dynamics_model,
            ship_model=ship_model,
            wind=wind,
            current=current,
            numerical_damping=numerical_damping,
            enable_obstacles=enable_obstacles,
            enable_variable_ship_size=enable_variable_ship_size,
            encounter_range=self.polar_max_range,
            eval=eval,
            scenario_types=scenario_types,
            scenario_probs=scenario_probs,
            obstacle_population=obstacle_population,
            obstacle_weights=obstacle_weights,
            scenario_obstacle_level=scenario_obstacle_level,
            ground_truth_classification=ground_truth_classification,
        )

        self.polar_scan_cached = None

        # Reward weights for polar COLAV shaping
        self.reward_weights = {
            "polar_distance": 0.7,
            "colreg_bearing": 5.0,
            # Crossing-right shaping in the OS->TS bearing frame.
            #   x   = bearing of TS as seen from OS (math convention, CCW+)
            #   R   = A * tanh(k * (x - e1))
            # Zero crossing (and max gradient) at x = e1, chosen as the
            # "most positive" end of the encounter range so that the agent
            # gets the strongest learning signal right at encounter entry.
            # The wrap discontinuity at x = +/- pi is intentional: reaching
            # +A via the wrap requires first traversing a long stretch of
            # negative reward, so the agent cannot exploit it.
            "crossing_right": 1.0,
            "crossing_right_steepness": 1.0,
            "crossing_right_zero": -0.34906585,  # -20deg
            # Soft gate that fades the crossing-right shaping after TS leaves
            # the forward sector. Multiplier is ~1 for |x| < pi/2 (TS in front
            # = encounter active) and ~0 for |x| > pi/2 (TS abeam/astern =
            # passed). Transition width is set by gate_steepness.
            "crossing_right_gate_steepness": 4.0,
        }

    # -------------------------
    # Polar scan geometry
    # -------------------------
    def _intersect_rays_with_segments(self, origins, ray_ends, seg_starts, seg_ends):
        """Vectorized ray-segment intersection distances."""
        r = ray_ends - origins
        s = seg_ends - seg_starts
        origin_to_starts = seg_starts - origins

        denom = r[:, np.newaxis, 0] * s[np.newaxis, :, 1] - r[:, np.newaxis, 1] * s[np.newaxis, :, 0]
        denom[np.abs(denom) < 1e-10] = np.nan

        t = (origin_to_starts[np.newaxis, :, 0] * s[np.newaxis, :, 1] -
             origin_to_starts[np.newaxis, :, 1] * s[np.newaxis, :, 0]) / denom
        u = (origin_to_starts[np.newaxis, :, 0] * r[:, np.newaxis, 1] -
             origin_to_starts[np.newaxis, :, 1] * r[:, np.newaxis, 0]) / denom

        valid = (t >= 0) & (t <= 1) & (u >= 0) & (u <= 1)
        dist_matrix = np.where(valid, t * self.polar_max_range, self.polar_max_range)
        return np.min(dist_matrix, axis=1)

    def _rebuild_polar_scan_geometries(self) -> None:
        """Prepare boundary and collision-entity geometry for vectorized ray casting."""
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

        ch, sh = np.cos(heading), np.sin(heading)
        dx = ch * self._polar_rel_dirs[:, 0] - sh * self._polar_rel_dirs[:, 1]
        dy = sh * self._polar_rel_dirs[:, 0] + ch * self._polar_rel_dirs[:, 1]

        ray_ends = np.stack([x0 + self.polar_max_range * dx,
                             y0 + self.polar_max_range * dy], axis=1)

        buffer = 50.0
        limit_sq = (self.polar_max_range + buffer) ** 2
        d0 = (self._boundary_starts[:, 0] - x0) ** 2 + (self._boundary_starts[:, 1] - y0) ** 2
        d1 = (self._boundary_ends[:, 0] - x0) ** 2 + (self._boundary_ends[:, 1] - y0) ** 2

        mask = (d0 < limit_sq) | (d1 < limit_sq)
        local_starts = self._boundary_starts[mask]
        local_ends = self._boundary_ends[mask]

        distances = np.full(self.polar_n_sectors, self.polar_max_range, dtype=np.float32)

        if len(local_starts) > 0:
            distances = self._intersect_rays_with_segments(
                ship_pos, ray_ends, local_starts, local_ends
            )

        if self._collision_entity_centers is not None and len(self._collision_entity_centers) > 0:
            d = ray_ends - ship_pos
            f = ship_pos - self._collision_entity_centers

            a = np.sum(d ** 2, axis=1)[:, np.newaxis]
            b = 2 * (d @ f.T)
            c = (np.sum(f ** 2, axis=1) - self._collision_entity_radii_sq)[np.newaxis, :]

            discriminant = b ** 2 - 4 * a * c
            hits = (discriminant >= 0)
            t = np.where(hits, (-b - np.sqrt(np.maximum(0, discriminant))) / (2 * a), 1.0)
            valid_t = np.where((t >= 0) & (t <= 1), t * self.polar_max_range, self.polar_max_range)
            circle_distances = np.min(valid_t, axis=1)
            distances = np.minimum(distances, circle_distances)

        dist = np.clip(distances / self.polar_max_range, 0.0, 1.0).astype(np.float32)
        closeness = 1.0 - dist
        scan = closeness * float(self.polar_obs_max)
        return np.clip(scan, 0.0, float(self.polar_obs_max)).astype(np.float32)

    # -----------------------
    # Hooks from base
    # -----------------------
    def _on_entities_stepped(self) -> None:
        """Rebuild polar scan geometries after entities move."""
        if self.use_polar_scan:
            if not self.phys_sim.state.terminated and not self.phys_sim.state.truncated:
                self._rebuild_polar_scan_geometries()
                self.polar_scan_cached = self._get_polar_scan()

    def _on_episode_reset(self) -> None:
        """Reset polar-specific state for new episode."""
        if getattr(self, "use_polar_scan", False):
            self.polar_scan_cached = None

        self._colreg_initial_heading = None
        self._crossing_left_initial_speed = None
        self._crossing_right_grav_point = None

        if getattr(self, "use_polar_scan", False):
            self._rebuild_polar_scan_geometries()

        self._overtaking_pass_reported = False
        self._os_speed_samples = []

    # -----------------------
    # Observation
    # -----------------------
    def _initialize_observation_space(self) -> gym.spaces.Box:
        """Extend observation space with polar scan sectors + CRI."""
        base_space = super()._initialize_observation_space()

        polar_low = np.zeros(self.polar_n_sectors, dtype=np.float32)
        polar_high = np.full(self.polar_n_sectors, self.polar_obs_max, dtype=np.float32)

        cri_low = np.zeros(1, dtype=np.float32)
        cri_high = np.ones(1, dtype=np.float32)

        extended_low = np.hstack([base_space.low, polar_low, cri_low])
        extended_high = np.hstack([base_space.high, polar_high, cri_high])

        if getattr(self, "include_scenario_one_hot", False):
            one_hot_low = np.zeros(NUM_SCENARIOS, dtype=np.float32)
            one_hot_high = np.ones(NUM_SCENARIOS, dtype=np.float32)
            extended_low = np.hstack([extended_low, one_hot_low])
            extended_high = np.hstack([extended_high, one_hot_high])

        return gym.spaces.Box(low=extended_low, high=extended_high, dtype=np.float32)

    def _get_obs(self) -> np.ndarray:
        """Extend observation with polar scan data and CRI scalar."""
        base_obs = super()._get_obs()

        if not getattr(self, "use_polar_scan", False):
            zeros_polar = np.zeros(self.polar_n_sectors, dtype=np.float32)
            zeros_cri = np.zeros(1, dtype=np.float32)
            obs = np.hstack([base_obs, zeros_polar, zeros_cri]).astype(np.float32)
            if getattr(self, "include_scenario_one_hot", False):
                obs = np.hstack([obs, np.zeros(NUM_SCENARIOS, dtype=np.float32)]).astype(np.float32)
            return obs

        scan = self.polar_scan_cached
        if scan is None:
            scan = self._get_polar_scan()
            self.polar_scan_cached = scan

        if not self.phys_sim.state.terminated and not self.phys_sim.state.truncated:
            self.scenario_classifier.classify_path_projected(
                self.ship_pos,
                self.phys_sim.state.heading,
                self.collision_entities,
            )

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

        if getattr(self, "include_scenario_one_hot", False):
            obs = np.hstack([obs, self.scenario_classifier.get_one_hot()]).astype(np.float32)

        return obs

    # -------------------
    # Reward
    # -------------------
    def _calculate_colav_reward(self) -> Tuple[float, bool]:
        """Polar COLAV reward: base path-following + bearing-based COLREG shaping."""
        base_reward, done = super()._calculate_colav_reward()
        reward = base_reward

        # Polar distance penalty
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
        self.ts_idx = clf.tracked_ts

        if clf.label != "crossing-left":
            self._crossing_left_initial_speed = None
        if clf.label != "crossing-right":
            self._crossing_right_grav_point = None

        _headon_term = 0.0
        _step_grav = 0.0
        if self.ts_idx is not None and self.ts_idx < len(self.collision_entities):
            ts_idx = self.ts_idx
            ts = self.collision_entities[ts_idx]
            os_pos = self.ship_pos
            os_heading = self.phys_sim.state.heading
            scenario = clf.label

            if scenario == "head-on" or scenario == "overtaking":
                ts_pos = ts.get_position()
                dx = float(ts_pos[0] - os_pos[0])
                dy = float(ts_pos[1] - os_pos[1])
                os_u = float(self.phys_sim.state.u)
                os_v = float(self.phys_sim.state.v)
                rel_bearing = (
                                      math.atan2(dy, dx) - os_heading + math.pi
                              ) % (2.0 * math.pi) - math.pi

                dist = ts.distance_to_point(os_pos)
                proximity = max(0.0, 1.0 - dist / self.polar_max_range)
                cri = self._compute_cri(os_pos, os_heading, os_u, os_v, ts)
                _sin_b = math.sin(rel_bearing)
                _headon_term = self.reward_weights["colreg_bearing"] * _sin_b * (1-cri)
                reward += _headon_term

                if scenario == "head-on":
                    side = ts.ts_side_from_os_perspective(os_pos, os_heading)
                    if ts.passed_by_os(os_pos, os_heading) and self.ts_passed != ts_idx:
                        self.ts_passed = ts_idx
                        if side == "port":
                            self._colreg_correct += 1
                        elif side == "starboard":
                            self._colreg_violations += 1

                        self._passing_snapshots.append({
                            "os_pos": np.copy(os_pos),
                            "ts_pos": ts_pos.copy(),
                            "dist": dist,
                            "scenario": scenario,
                        })

            elif scenario == "crossing-right":
                ts_pos = ts.get_position()

                # x = bearing of TS as seen from OS, in OS body frame
                # (math convention, CCW positive). x in (-pi, pi]:
                #   x =  0     -> TS dead ahead
                #   x = +pi/2  -> TS on OS port
                #   x = -pi/2  -> TS on OS starboard
                #   x = +/-pi  -> TS astern
                dx = float(ts_pos[0] - os_pos[0])
                dy = float(ts_pos[1] - os_pos[1])
                x = (
                    math.atan2(dy, dx) - os_heading + math.pi
                ) % (2.0 * math.pi) - math.pi

                # R(x) = A * tanh(k * (x - e1)).
                # Zero crossing (and max gradient) at x = e1, set to the
                # "most positive" end of the encounter range so the unlearned
                # agent enters at the steepest part of the curve. The wrap
                # discontinuity at x = +/- pi is intentional: reaching +A
                # via the wrap requires traversing a long stretch of negative
                # reward first, so it cannot be exploited.
                k = float(self.reward_weights["crossing_right_steepness"])
                e1 = float(self.reward_weights["crossing_right_zero"])
                k_gate = float(self.reward_weights["crossing_right_gate_steepness"])

                # Front/back-sector soft gate on |x|: ~1 when TS is in the
                # forward sector (|x| < pi/2 = encounter active), ~0 once
                # TS is abeam/astern (|x| > pi/2 = passed). Fades the
                # shaping so path-following reward dominates after the pass.
                #gate = 0.5 * (1.0 - math.tanh(k_gate * (abs(x) - 0.5 * math.pi)))

                crossing_right_signal = self.reward_weights["crossing_right"] * math.tanh(k * (x - e1))
                reward += crossing_right_signal

                # COLREG violation tracking
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

                    # Re-lock the scenario to "path-following" now that the
                    # TS has been passed (|rel_bearing| > pi/2). This forces
                    # the MultiPolicySingleAgent to switch to the
                    # path-following policy for the recovery phase, while
                    # the TS is still in forget range. The lock clears
                    # automatically via scenario_classifier.reset() when the
                    # TS leaves forget range (see scenario_builder.on_step).
                    # We deliberately do NOT release to geometric mode here:
                    # post-pass geometry can be classified as "overtaking"
                    # (OS abaft TS beam) which would route to the wrong
                    # policy.
                    if self.ground_truth_classification:
                        clf.set_episode_scenario("path-following")
        return reward, done

    # -----------------------
    # Rendering (polar-specific)
    # -----------------------
    def _initialize_rendering(self) -> None:
        super()._initialize_rendering()

        self.polar_beams = []
        if self.use_polar_scan:
            for _ in range(self.polar_n_sectors):
                beam_line, = self.ax.plot(
                    [], [], color='orange', alpha=0.3, linewidth=1,
                )
                self.polar_beams.append(beam_line)

    def _render_subclass(self, heading: float) -> None:
        """Update polar beam rendering."""
        if self.use_polar_scan and getattr(self, "polar_beams", None):
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

    def _blit_subclass(self) -> None:
        """Blit polar beam artists."""
        if self.use_polar_scan and getattr(self, "polar_beams", None):
            for beam in self.polar_beams:
                self.ax.draw_artist(beam)

    # -----------------------
    # Evaluation
    # -----------------------
    def plot_min_distance_PDF(self, save_path: str = "min_distance_PDF.png") -> None:
        """Plot probability density of minimum center-to-center distance
        to collision entities across non-collision evaluation episodes."""
        print("plot min dist PDF")

        if not self._current_episode_collided and self._current_episode_min_dist < float("inf"):
            self._episode_min_distances.append(self._current_episode_min_dist)

        distances = np.array(self._episode_min_distances)
        if len(distances) == 0:
            print("No non-collision episodes to plot min-distance png for.")
            return

        fig, ax = plt.subplots(figsize=(10, 6))

        n_bins = max(10, len(distances) // 3)
        ax.hist(distances, bins=n_bins, density=True, alpha=0.6, color="steelblue",
                edgecolor="white", label="Histogram (density)")

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

        row1_y = -0.06
        if os_r > 0:
            ax.annotate("", xy=(os_r, row1_y), xytext=(0, row1_y),
                         arrowprops=dict(**arrow_kwargs, color="red"),
                         xycoords=arrow_coords, textcoords=arrow_coords, clip_on=False)
            ax.text(os_r / 2, row1_y - 0.02, f"OS radius = {os_r:.0f}m",
                    transform=ax.get_xaxis_transform(), color="red",
                    clip_on=False, **text_kwargs)

        row2_y = row1_y - 0.08
        if variable_ts:
            ax.annotate("", xy=(os_r + ts_r_min, row2_y), xytext=(os_r, row2_y),
                         arrowprops=dict(**arrow_kwargs, color="orange"),
                         xycoords=arrow_coords, textcoords=arrow_coords, clip_on=False)
            ax.text(os_r + ts_r_min / 2, row2_y - 0.02,
                    f"min TS radius = {ts_r_min:.0f}m",
                    transform=ax.get_xaxis_transform(), color="orange",
                    clip_on=False, **text_kwargs)
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

    def plot_scenario_confusion_matrix(
        self,
        save_path: str = "scenario_confusion_matrix.png",
    ) -> None:
        """Plot a confusion matrix over encounters aggregated across episodes.

        Rows = builder ground-truth scenario, columns = geometric classifier
        prediction at encounter entry. One entry per encounter (multiple
        encounters may occur per episode in step-level generation).
        """
        print("plot scenario confusion")
        # Flush current episode's encounters if the run ended without another
        # env_specific_reset() being called.
        encounters = list(self._all_encounters)
        if self._episode_encounters:
            encounters.extend(self._episode_encounters)

        if not encounters:
            print("No encounters to plot scenario confusion matrix for.")
            return

        labels = list(EXTENDED_SCENARIO_LABELS)
        label_to_idx = {lbl: i for i, lbl in enumerate(labels)}
        n = len(labels)

        matrix = np.zeros((n, n), dtype=np.int32)
        for enc in encounters:
            gt = enc.get("gt")
            pred = enc.get("pred")
            if gt not in label_to_idx or pred not in label_to_idx:
                continue
            matrix[label_to_idx[gt], label_to_idx[pred]] += 1

        total = int(matrix.sum())
        correct = int(np.trace(matrix))
        accuracy = (correct / total * 100.0) if total else 0.0

        fig, ax = plt.subplots(figsize=(8, 7))
        im = ax.imshow(matrix, cmap="Blues", aspect="equal")

        ax.set_xticks(np.arange(n))
        ax.set_yticks(np.arange(n))
        ax.set_xticklabels(labels, rotation=30, ha="right")
        ax.set_yticklabels(labels)
        ax.set_xlabel("Geometric prediction (at encounter entry)")
        ax.set_ylabel("Ground truth (builder scenario)")
        ax.set_title(
            f"Scenario confusion matrix — {total} encounters, "
            f"accuracy {accuracy:.1f}% ({correct}/{total})"
        )

        # Overlay counts. Use white text on dark cells for readability.
        thresh = matrix.max() / 2.0 if matrix.max() > 0 else 0.5
        for i in range(n):
            for j in range(n):
                val = int(matrix[i, j])
                color = "white" if val > thresh else "black"
                ax.text(
                    j, i, str(val),
                    ha="center", va="center",
                    color=color, fontsize=10, fontweight="bold",
                )

        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Encounter count")
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Scenario confusion matrix saved as '{save_path}'")
        plt.close(fig)


# Backwards-compatible alias
PySimColavEnv = PySimColavPolarEnv
