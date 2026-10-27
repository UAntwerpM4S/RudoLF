"""
Polar rangefinder collision avoidance environment.

Extends PySimColavBaseEnv with a polar rangefinder observation and
COLREG-aware reward shaping for the polar/rangefinder approach.
"""

import math
import logging
import matplotlib
import numpy as np
import gymnasium as gym

from typing import Optional, Tuple
from fw.ships.ship_factory import ShipModel
from fw.dynamics.dynamics_model import Model
from fw.environments.py_sim_colav_base import PySimColavBaseEnv
from fw.collision_avoidance.scenario_classifier import NUM_SCENARIOS

EPSILON = 1e-9


class PySimColavPolarEnv(PySimColavBaseEnv):
    """
    Collision avoidance environment using polar rangefinder observations.

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
            overall_contour: Optional[str] = None,
            depth_contour: Optional[str] = None,
            trajectory: Optional[str] = None,
            dynamics_model: Optional[Model] = Model.FOSSEN,
            ship_model: Optional[ShipModel] = ShipModel.MYZAKO_DEFAULT,
            wind: bool = False,
            current: bool = False,
            ugent_surrogate_config: Optional[dict] = None,
            enable_obstacles: bool = True,
            enable_variable_ship_size: bool = False,
            evaluate: bool = False,
            include_scenario_one_hot: bool = False,
            # Scenario configuration (passed to base)
            scenario_types: Optional[list] = None,
            scenario_probs: Optional[list] = None,
            obstacle_population: Optional[list] = None,
            obstacle_weights: Optional[list] = None,
            scenario_obstacle_level: Optional[str] = None,
            ground_truth_classification: bool = True,
            # Focused-encounter training configuration
            focused_encounter_mode: bool = True,
            train_encounter_prob: float = 1.0,
            train_encounter_cooldown: int = 150,
            eval_encounter_prob: float = 0.004,
            eval_encounter_cooldown: int = 500,
            post_encounter_steps: int = 150,
            pathfollowing_encounter_steps: int = 50,
            skip_tail_segments: int = 5,
            random_segment_start: bool = True,
            minimal_path_render: bool = False,
            **kwargs,
    ) -> None:
        """
        Initializes the polar collision avoidance environment.

        Configures the polar rangefinder, COLREG-aware reward shaping, focused
        encounter training, and scenario generation before initializing the base
        collision avoidance environment.

        Args:
            render_mode: Rendering mode for the environment.
            time_step: Simulation time step in seconds.
            max_steps: Maximum number of simulation steps per episode.
            verbose: Enables verbose logging if True.
            ship_pos: Initial own-ship position. Randomized if None.
            target_pos: Target position. Randomized if None.
            dynamics_model: Vessel dynamics model.
            ship_model: Ship specification model.
            wind: Enables wind disturbances.
            current: Enables current disturbances.
            ugent_surrogate_config: Configuration for the UGent surrogate
                (Model.UGENT_SURROGATE), e.g. {"ukc_percent": 363}.
            enable_obstacles: Enables obstacle generation.
            enable_variable_ship_size: Enables randomized ship sizes.
            evaluate: Enables evaluation mode.
            include_scenario_one_hot: Appends a one-hot scenario encoding to the
                observation.
            scenario_types: Available encounter scenario types.
            scenario_probs: Sampling probabilities for scenario types.
            obstacle_population: Available obstacle density levels.
            obstacle_weights: Sampling weights for obstacle densities.
            scenario_obstacle_level: Obstacle generation level.
            ground_truth_classification: Uses ground-truth encounter labels instead
                of classifier predictions.
            focused_encounter_mode: Enables encounter-focused training episodes.
            train_encounter_prob: Encounter probability during training.
            train_encounter_cooldown: Minimum steps between training encounters.
            eval_encounter_prob: Encounter probability during evaluation.
            eval_encounter_cooldown: Minimum steps between evaluation encounters.
            post_encounter_steps: Steps after an encounter before terminating.
            pathfollowing_encounter_steps: Initial path-following steps before an
                encounter.
            skip_tail_segments: Number of path segments excluded from encounter
                placement.
            random_segment_start: Randomizes the initial path segment.
            minimal_path_render: Enables simplified path rendering.
            **kwargs: Reserved for future compatibility.
        """

        # Polar scan configuration
        self.polar_scan_cached = None
        self.polar_n_sectors = 54
        self.polar_fov = np.radians(270)
        self.polar_max_range = 750.0
        self.polar_obs_max = 0.1 * self.polar_max_range / 250
        self.distance_penalty_start = 100
        self.distance_penalty_peak = 40
        self.distance_penalty_max = 5.0

        # overtaking TS speed
        self.overtaking_ts_v_min = 1.0
        self.overtaking_ts_v_max = 1.7

        self.include_scenario_one_hot = bool(include_scenario_one_hot)

        # focused encounter (path following -> encounter -> path following --> terminate)
        # to produce shorter episodes with more equal step distribution across encounter/path-following
        self.focused_encounter_mode = bool(focused_encounter_mode)
        self.train_encounter_prob = float(train_encounter_prob)
        self.train_encounter_cooldown = int(train_encounter_cooldown)
        self.eval_encounter_prob = float(eval_encounter_prob)
        self.eval_encounter_cooldown = int(eval_encounter_cooldown)
        self.post_encounter_steps = int(post_encounter_steps)
        self.pathfollowing_encounter_steps = int(pathfollowing_encounter_steps)
        self.skip_tail_segments = int(skip_tail_segments)
        self.random_segment_start = bool(random_segment_start)


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
            0.0, 0.0, 0.33, 0.33, 0.34, 0.0 # "path-following", "static", "overtaking", "head-on", "crossing-right", "crossing-left"
        ]

        # Call base init (encounter_range = polar_max_range)
        super().__init__(
            render_mode=render_mode,
            time_step=time_step,
            max_steps=max_steps,
            verbose=verbose,
            ship_pos=ship_pos,
            target_pos=target_pos,
            overall_contour=overall_contour,
            depth_contour=depth_contour,
            trajectory=trajectory,
            dynamics_model=dynamics_model,
            ship_model=ship_model,
            wind=wind,
            current=current,
            ugent_surrogate_config=ugent_surrogate_config,
            enable_obstacles=enable_obstacles,
            enable_variable_ship_size=enable_variable_ship_size,
            encounter_range=self.polar_max_range,
            evaluate=evaluate,
            scenario_types=scenario_types,
            scenario_probs=scenario_probs,
            obstacle_population=obstacle_population,
            obstacle_weights=obstacle_weights,
            scenario_obstacle_level=scenario_obstacle_level,
            ground_truth_classification=ground_truth_classification,
            minimal_path_render=minimal_path_render,
        )

        self.polar_scan_cached = None
        self.enable_reward_diag = False

        # Reward weights for polar COLAV shaping
        self.reward_weights = {
            "polar_distance_colreg": 0.65,
            "colreg_bearing": 1.5,
            "crossing-right": 0.53125,
            "head-on": 0.75,
            "overtaking": 0.53125,
        }

        # Estimated peak of base_reward
        self._base_reward_peak_est = 0.9

    # -------------------------
    # Evaluation-mode overrides
    # -------------------------
    def set_evaluation_parameters(self) -> None:
        """
        Flip on the eval-only realism overrides for trained-model evaluation.

        Called by AgentManager.visualize_trained_agent before running episodes.
        Overrides set here take effect on the next reset/spawn. They are read
        by the scenario builder (lateral spawn) and the base env step loop
        (head-on dodge). Training runs leave these at their safe defaults
        (set in PySimColavBaseEnv.__init__) and are unaffected.
        """

        self.eval = True

        # overtaking TS speed
        self.overtaking_ts_v_min = 1.0
        self.overtaking_ts_v_max = 1.0

        # Head-on TS speed
        self.headon_ts_v_min = self.os_nominal_speed
        self.headon_ts_v_max = self.os_nominal_speed

        # Let OS always start from the original position
        self.random_segment_start = False

        # Single-encounter mode wiring (read by scenario_builder).
        # self.single_encounter_per_episode = True
        # self.single_encounter_target_segment_index = 15
        # self.single_encounter_zoom_halfwidth = 1450

    # -------------------------
    # Polar scan geometry
    # -------------------------
    def _intersect_rays_with_segments(self, origins, ray_ends, seg_starts, seg_ends):
        """
        Vectorized ray-segment intersection distances.
        """

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
        """
        Prepare boundary and collision-entity geometry for vectorized ray casting.
        """

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
        """
        Computes the current polar rangefinder scan.

        Performs vectorized ray casting against the environment boundaries and
        collision entities to generate the normalized polar scan observation.

        Returns:
            A normalized polar scan with one value per sector.
        """

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
        """
        Rebuild polar scan geometries after entities move.
        """

        if not self.phys_sim.state.terminated and not self.phys_sim.state.truncated:
            self._rebuild_polar_scan_geometries()
            self.polar_scan_cached = self._get_polar_scan()

        self._update_passing_detection()

    def _update_passing_detection(self) -> None:
        """
        Track TS passing events and record encounter snapshots.

        TS-frame sign-flip on lateral (crossing-right) or longitudinal
        (head-on) offset. Linearly interpolate the sub-step crossing point
        so the dotted snapshot line lies on the TS track.
        """

        clf = self.scenario_classifier
        ts_idx = clf.tracked_ts

        if (
            ts_idx is None
            or ts_idx >= len(self.collision_entities)
            or clf.label not in ("head-on", "crossing-right", "overtaking")
        ):
            self._prev_passing_signal = None
            self._prev_passing_ts_idx = None
            return

        ts = self.collision_entities[ts_idx]
        os_pos_now = self.ship_pos
        ts_pos_now = ts.get_position()
        signal = ts.passing_signal(os_pos_now, clf.label)

        # Encounter-start snapshot (first step we observe this TS in a
        # tracked encounter scenario).
        if self._prev_passing_ts_idx != ts_idx:
            self._passing_snapshots.append({
                "os_pos": np.asarray(os_pos_now, dtype=np.float32),
                "ts_pos": np.asarray(ts_pos_now, dtype=np.float32),
                "dist": float(np.linalg.norm(os_pos_now - ts_pos_now)),
                "scenario": clf.label,
                "kind": "start",
            })

        prev = self._prev_passing_signal
        if (
            prev is not None
            and self._prev_passing_ts_idx == ts_idx
            and prev * signal < 0.0
            and self.ts_passed != ts_idx
        ):
            t = prev / (prev - signal)
            os_at = self._prev_os_pos + t * (os_pos_now - self._prev_os_pos)
            ts_at = self._prev_ts_pos + t * (ts_pos_now - self._prev_ts_pos)
            self.ts_passed = ts_idx
            self._current_encounter_passed = True
            side = ts.ts_side_from_os_perspective(os_pos_now, self.phys_sim.state.heading)
            if side == "port":
                self._colreg_correct += 1
            elif side == "starboard":
                self._colreg_violations += 1
            self._passing_snapshots.append({
                "os_pos": np.asarray(os_at, dtype=np.float32),
                "ts_pos": np.asarray(ts_at, dtype=np.float32),
                "dist": float(np.linalg.norm(os_at - ts_at)),
                "scenario": clf.label,
                "kind": "passed",
            })
        self._prev_passing_signal = signal
        self._prev_passing_ts_idx = ts_idx
        self._prev_os_pos = np.copy(os_pos_now)
        self._prev_ts_pos = np.copy(ts_pos_now)

    def _on_episode_reset(self) -> None:
        """
        Reset polar-specific state for new episode.
        """

        self.polar_scan_cached = None
        self._rebuild_polar_scan_geometries()

        self._overtaking_pass_reported = False
        self._os_speed_samples = []

    # -----------------------
    # Observation
    # -----------------------
    def _initialize_observation_space(self) -> gym.spaces.Box:
        """
        Extend observation space with polar scan sectors + tanh-TCPA scalar.
        """

        base_space = super()._initialize_observation_space()

        polar_low = np.zeros(self.polar_n_sectors, dtype=np.float32)
        polar_high = np.full(self.polar_n_sectors, self.polar_obs_max, dtype=np.float32)

        # tanh(TCPA / cri_T) is bounded in (-1, +1).
        tcpa_low = -np.ones(1, dtype=np.float32)
        tcpa_high = np.ones(1, dtype=np.float32)

        extended_low = np.hstack([base_space.low, polar_low, tcpa_low])
        extended_high = np.hstack([base_space.high, polar_high, tcpa_high])

        if getattr(self, "include_scenario_one_hot", False):
            one_hot_low = np.zeros(NUM_SCENARIOS, dtype=np.float32)
            one_hot_high = np.ones(NUM_SCENARIOS, dtype=np.float32)
            extended_low = np.hstack([extended_low, one_hot_low])
            extended_high = np.hstack([extended_high, one_hot_high])

        return gym.spaces.Box(low=extended_low, high=extended_high, dtype=np.float32)

    def _get_obs(self) -> np.ndarray:
        """
        Extend observation with polar scan data and a TCPA phase scalar.
        """

        base_obs = super()._get_obs()

        clf = self.scenario_classifier

        if not self.phys_sim.state.terminated and not self.phys_sim.state.truncated:
            clf.classify_path_projected(
                self.ship_pos,
                self.phys_sim.state.heading,
                self.collision_entities,
            )

        scan = self.polar_scan_cached
        if scan is None:
            scan = self._get_polar_scan()
            self.polar_scan_cached = scan

        # Base path-following without radial distances
        if clf.label == "path-following":
            scan = np.zeros(self.polar_n_sectors, dtype=np.float32)

        # Encounter phase as a single scalar
        #   ~+1  TS before CPA, 0 at CPA, ~-1 TS already past CPA
        tcpa_norm = 0.0
        if clf.tracked_ts is not None and clf.tracked_ts < len(self.collision_entities):
            ts = self.collision_entities[clf.tracked_ts]
            os_pos = self.ship_pos
            os_heading = self.phys_sim.state.heading
            os_u = float(self.phys_sim.state.u)
            os_v = float(self.phys_sim.state.v)
            _, tcpa, _ = self._compute_cri(os_pos, os_heading, os_u, os_v, ts)
            tcpa_norm = float(math.tanh(tcpa / max(self.cri_T, 1e-6)))

        obs = np.hstack([
            base_obs,
            scan.astype(np.float32),
            np.array([tcpa_norm], dtype=np.float32),
        ]).astype(np.float32)

        if getattr(self, "include_scenario_one_hot", False):
            obs = np.hstack([obs, self.scenario_classifier.get_one_hot()]).astype(np.float32)

        return obs

    # -------------------
    # Reward
    # -------------------
    def _calculate_colav_reward(self) -> Tuple[float, bool]:
        """
        Polar COLAV reward: base path-following + bearing-based COLREG shaping.
        """

        base_reward, done = super()._calculate_colav_reward()
        reward = base_reward

        clf = self.scenario_classifier
        self.ts_idx = clf.tracked_ts

        g_tcpa = 1.0
        cri = 0.0
        if (
            self.ts_idx is not None
            and self.ts_idx < len(self.collision_entities)
            and clf.label in ("crossing-right", "head-on", "overtaking")
        ):
            ts = self.collision_entities[self.ts_idx]
            os_u = float(self.phys_sim.state.u)
            os_v = float(self.phys_sim.state.v)
            cri, tcpa, _ = self._compute_cri(
                self.ship_pos, self.phys_sim.state.heading, os_u, os_v, ts,
            )
            k_tcpa = 0.15
            g_tcpa = 0.5 * (1.0 + math.tanh(k_tcpa * tcpa))

        _step_polar = 0.0
        if (
            self.polar_scan_cached is not None
            and clf.label != "path-following"
        ):
            max_close = float(np.max(self.polar_scan_cached))
            min_dist_n = 1.0 - (max_close / max(float(self.polar_obs_max), 1e-6))

            d_ramp_start = self.distance_penalty_start
            ramp_max_penalty = self.distance_penalty_max
            min_dist = min_dist_n * self.polar_max_range

            if min_dist < d_ramp_start:
                distance_penalty_peak = self.distance_penalty_peak
                a = -2.6
                ramp_t = ((min_dist - distance_penalty_peak) /
                          max(float(d_ramp_start - distance_penalty_peak), EPSILON))
                ramp_t = float(np.clip(ramp_t, 0.0, 1.0))
                exp_t = (np.exp(a * ramp_t) - 1.0) / (np.exp(a) - 1.0)
                ramp_penalty = ramp_max_penalty * (1.0 - exp_t)
                polar_weight = float(self.reward_weights["polar_distance_colreg"])
                _step_polar = polar_weight * ramp_penalty
                reward -= _step_polar

        _headon_term = 0.0
        if self.ts_idx is not None and self.ts_idx < len(self.collision_entities):
            ts_idx = self.ts_idx
            ts = self.collision_entities[ts_idx]
            os_pos = self.ship_pos
            os_heading = self.phys_sim.state.heading
            scenario = clf.label

            if scenario == "head-on":
                ts_pos = ts.get_position()
                dx = float(ts_pos[0] - os_pos[0])
                dy = float(ts_pos[1] - os_pos[1])
                beta = (
                    math.atan2(dy, dx) - os_heading + math.pi
                ) % (2.0 * math.pi) - math.pi
                # Same shape as crossing-right; (1 + (1 - cri)) boosts the
                # signal magnitude when the encounter is still low-risk so
                # earlier turning is rewarded more than late turning.
                k = 1
                _headon_term = (
                    self.reward_weights["head-on"]
                    * math.tanh(k * beta)
                    * (1 + (1 - cri))
                    * g_tcpa
                )
                reward += _headon_term

            elif scenario == "crossing-right":
                ts_pos = ts.get_position()
                dx = float(ts_pos[0] - os_pos[0])
                dy = float(ts_pos[1] - os_pos[1])
                beta = (
                    math.atan2(dy, dx) - os_heading + math.pi
                ) % (2.0 * math.pi) - math.pi

                # R(x) = A * tanh(k * (x - beta_start)).
                # Zero crossing (and max gradient) at encounter relative bearing
                k = 1
                beta_start = -0.34906585 # -20deg

                crossing_right_signal = (
                    self.reward_weights["crossing-right"]
                    * math.tanh(k * (beta - beta_start))
                    * g_tcpa
                )
                reward += crossing_right_signal

            elif scenario == "overtaking":
                ts_pos = ts.get_position()
                dx = float(ts_pos[0] - os_pos[0])
                dy = float(ts_pos[1] - os_pos[1])
                beta = (
                               math.atan2(dy, dx) - os_heading + math.pi
                       ) % (2.0 * math.pi) - math.pi

                k = 1
                _overtaking_term = (
                        self.reward_weights["overtaking"]
                        * math.tanh(k * beta)
                        * g_tcpa
                )
                reward += _overtaking_term


        # Post-encounter base-reward boost so that the
        # phase-2 base reward reaches same order of magnitude as the phase-1 reward
        # (base + scenario-specific directional signal) at saturation.
        if clf.label != "path-following":
            base_peak = float(self._base_reward_peak_est)
            cr_w = float(self.reward_weights[clf.label])
            phase2_multiplier = 1.0 + cr_w / max(base_peak, 1e-6)
            reward+= (phase2_multiplier - 1.0) * (1.0 - g_tcpa) * base_reward

        return reward, done

    # -----------------------
    # Rendering (polar-specific)
    # -----------------------
    def _initialize_rendering(self) -> None:
        """
        Initializes polar-specific rendering.

        Creates polar beam artists, configures the plot appearance, and applies
        evaluation-specific camera settings when required.
        """

        super()._initialize_rendering()

        self.polar_beams = []
        for _ in range(self.polar_n_sectors):
            beam_line, = self.ax.plot(
                [], [], color='orange', alpha=0.3, linewidth=1,
            )
            beam_line.set_animated(True)
            self.polar_beams.append(beam_line)

        # Override the legend with a curated, fixed set of entries (proxy
        # artists, so they don't depend on which artists got auto-labeled).
        from matplotlib.lines import Line2D
        legend_handles = [
            Line2D([0], [0], marker='o', color='none', markerfacecolor='blue',
                   markersize=8, label='Own ship'),
            Line2D([0], [0], marker='o', color='none', markerfacecolor='red',
                   markersize=8, label='Target ship'),
            Line2D([0], [0], color='green', alpha=0.2, linewidth=1.0,
                   label='Path'),
            Line2D([0], [0], color='blue', linewidth=1.5, label='OS trajectory'),
            Line2D([0], [0], color='red', linewidth=1.5, linestyle='--',
                   label='TS trajectory'),
        ]
        self.ax.legend(handles=legend_handles, loc='best', fontsize=14)

        # Paper-ready axis styling: titled, larger fonts, km on both axes.
        from matplotlib.ticker import FuncFormatter
        self.ax.set_title(
            "Ship Trajectory",
            fontsize=22,
        )
        self.ax.set_xlabel("X position [km]", fontsize=18)
        self.ax.set_ylabel("Y position [km]", fontsize=18)
        km_formatter = FuncFormatter(lambda v, _pos: f"{v / 1000:g}")
        self.ax.xaxis.set_major_formatter(km_formatter)
        self.ax.yaxis.set_major_formatter(km_formatter)
        self.ax.tick_params(axis="both", which="major", labelsize=16)

        # Single-encounter eval: center the camera on the dormant TS so
        # multiple agents share an identical static frame for overlay
        # comparison. Only the OS trajectory changes between runs.
        # Zoom is controlled by `single_encounter_zoom_halfwidth` (m).
        if (
            getattr(self, "single_encounter_per_episode", False)
            and getattr(self, "_single_encounter_ts_spawn_pos", None) is not None
        ):
            ts_pos = self._single_encounter_ts_spawn_pos
            half_x = float(getattr(self, "single_encounter_zoom_halfwidth", 600.0))
            half_y = half_x * (9.0 / 16.0)
            self.ax.set_xlim(float(ts_pos[0]) - half_x, float(ts_pos[0]) + half_x)
            self.ax.set_ylim(float(ts_pos[1]) - half_y, float(ts_pos[1]) + half_y)
            self.ax.set_aspect("equal", adjustable="box")
            self.ax.autoscale(False)

    def _render_subclass(self, heading: float) -> None:
        """
        Update polar beam rendering.
        """

        if getattr(self, "polar_beams", None):
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
        """
        Blit polar beam artists.
        """

        if getattr(self, "polar_beams", None):
            for beam in self.polar_beams:
                self.ax.draw_artist(beam)

    # -----------------------
    # Evaluation
    # -----------------------
    def plot_min_distance_stats(
        self,
        pdf_save_path: str = "min_distance_PDF.png",
        boxplot_save_path: str = "min_distance_boxplots.png",
        include_banks: bool = True,
    ) -> None:
        """
        Plots minimum-distance evaluation statistics.

        Args:
            pdf_save_path: Output path for the probability density plot.
            boxplot_save_path: Output path for the boxplot figure.
            include_banks: Includes bank distances in the statistics if True.
        """

        from fw.collision_avoidance.colav_statistics import plot_min_distance_stats
        plot_min_distance_stats(
            self, pdf_save_path, boxplot_save_path, include_banks=include_banks
        )

    def plot_scenario_confusion_matrix(
        self,
        save_path: str = "scenario_confusion_matrix.png",
    ) -> None:
        """
        Plots the encounter scenario confusion matrix.

        Args:
            save_path: Output path for the generated figure.
        """

        from fw.collision_avoidance.colav_statistics import plot_scenario_confusion_matrix
        plot_scenario_confusion_matrix(self, save_path)


# Backwards-compatible alias
PySimColavEnv = PySimColavPolarEnv
