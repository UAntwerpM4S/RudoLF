"""
Base environment for collision avoidance scenarios: scenario construction,
entity advancement along checkpoints and collision detection. Subclasses add
observations (e.g. polar rangefinder, replanner) and reward shaping.

Class hierarchy:
    PySimBaseEnv (path following; independent of PySimEnv, not kept in sync)
      -> PySimColavBaseEnv (scenario + entity management)
          -> PySimColavPolarEnv (rangefinder approach)
          -> PySimColavPlannerEnv (replanner approach)
"""

import math
import logging
from collections import deque
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

matplotlib.rcParams['figure.raise_window'] = False

from shapely.geometry import Point
from typing import Optional, Tuple, Dict
from fw.ships.ship_factory import ShipModel
from fw.dynamics.dynamics_model import Model
from fw.environments.py_sim_base_env import PySimBaseEnv
from fw.collision_avoidance.scenario_builder import ScenarioBuilder
from fw.collision_avoidance.scenario_classifier import ScenarioClassifier

logger = logging.getLogger(__name__)


# -----------------------
# Module-level constants
# -----------------------
CHECKPOINTS_DISTANCE = 350
MIN_GRID_POS = -11700
MAX_GRID_POS = 14500

SUCCESS_REWARD = 50.0
CHECKPOINT_AREA_SIZE = 1.0
TARGET_AREA_SIZE = 1.0

MAX_FIG_WIDTH = 1200
MAX_FIG_HEIGHT = 900
DPI = 100

PERPENDICULAR_LINE_LENGTH = 50.0
PERPENDICULAR_LINE_PROXIMITY = 2.0
HEADING_CHANGE_THRESHOLD = np.pi / 2.0
CROSS_TRACK_TERMINATION_MULTIPLIER = 2.0
EPSILON = 1e-9



class PySimTowingTankColavBaseEnv(PySimBaseEnv):
    """
    Base collision avoidance environment. Extends PySimBaseEnv with scenario
    generation (obstacles, TS encounters), entity advancement along checkpoints,
    bank and entity collision detection, scenario classification with COLREG
    tracking, and CRI (Collision Risk Index) computation.

    Subclasses must override:
        _calculate_colav_reward() -> (float, bool)

    Subclasses may override:
        _on_entities_stepped() - post-entity-step hook
        _initialize_observation_space() - extend observation space
        _get_obs() - extend observation vector
        _on_episode_reset() - additional per-episode reset logic
    """

    # Max past encounters / episode minimum distances kept for evaluation
    # statistics (oldest dropped first).
    METRIC_HISTORY_MAXLEN: int = 10_000

    # True for envs that track one target ship through the classifier (ts_idx);
    # only there does the ScenarioBuilder despawn a dynamic TS that has passed.
    # An explicit flag, because hasattr(env, "ts_idx") depends on whether a
    # reward call already created the attribute.
    TRACKS_TARGET_SHIP: bool = False


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
            minimal_path_render: bool = False,
            ugent_surrogate_config: Optional[dict] = None,
            ugent_b01_surrogate_config: Optional[dict] = None,
            # COLAV configuration
            enable_obstacles: bool = True,
            enable_variable_ship_size: bool = False,
            encounter_range: float = 1000.0,
            evaluate: bool = False,
            # Scenario type configuration
            scenario_types: Optional[list] = ["head-on"],
            scenario_probs: Optional[list] = [1.0],
            scenario_obstacle_level: Optional[str] = "episode",
            # Obstacle population configuration
            obstacle_population: Optional[list] = None,
            obstacle_weights: Optional[list] = None,
            # True: classifier locked to the builder's ground-truth label (geometric
            # classification only computed for metrics). False: pure geometric
            # classification, so its error rate can be measured.
            ground_truth_classification: bool = True,
    ) -> None:
        """
        Args:
            render_mode: Rendering mode for the environment.
            time_step: Simulation time step in seconds.
            max_steps: Maximum number of simulation steps per episode.
            verbose: Enables verbose logging if True.
            ship_pos: Initial own-ship position. Randomized if None.
            target_pos: Target position. Randomized if None.
            overall_contour: Overall contour file.
            depth_contour: Depth contour file.
            trajectory: Trajectory file.
            dynamics_model: Vessel dynamics model.
            ship_model: Ship specification model.
            wind: Enables wind disturbances.
            current: Enables current disturbances.
            minimal_path_render: Enables simplified path rendering.
            ugent_surrogate_config: Configuration for the UGent surrogate
                (Model.UGENT_SURROGATE), e.g. {"ukc_percent": 363}.
            ugent_b01_surrogate_config: Configuration for the UGent B01 model-scale
                surrogate (Model.UGENT_B01_SURROGATE), e.g. {"ukc_percent": 379}.
            enable_obstacles: Enables obstacle generation.
            enable_variable_ship_size: Enables randomized target ship sizes.
            encounter_range: Distance within which encounters become active.
            evaluate: Enables evaluation mode.
            scenario_types: Available encounter scenario types.
            scenario_probs: Sampling probabilities for scenario types.
            scenario_obstacle_level: "episode" (default) or "step": when obstacles
                are generated.
            obstacle_population: Available obstacle counts.
            obstacle_weights: Sampling weights for obstacle counts.
            ground_truth_classification: Use scenario-builder labels instead of
                geometric classification for reward shaping and the one-hot.
        """

        self.ground_truth_classification = bool(ground_truth_classification)

        # Detection range for scenario activation, dodge triggers and
        # classification; subclasses may set it to their sensor range.
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
        self.headon_ts_v_min = self.os_nominal_speed
        self.headon_ts_v_max = self.os_nominal_speed
        self.crossing_ts_v_min = 1.0
        self.crossing_ts_v_max = 5.0
        self.crossing_activation_margin = 0.0
        self.crossing_speed_factor = 1.0
        self.own_ship_radius = 0.0

        # Evaluation / tracking
        self.eval = bool(evaluate)
        self.min_distance = float("inf")

        # Per-episode minimum distances for evaluation statistics. Capped because
        # every training worker (a deep copy) also appends to its unused copy;
        # the cap is far above what an evaluation run produces.
        self._episode_min_distances: deque = deque(maxlen=self.METRIC_HISTORY_MAXLEN)
        self._current_episode_min_dist: float = float("inf")
        self._current_episode_collided: bool = False

        # Per-encounter tracking. Each encounter adds a record to
        # _episode_encounters / _all_encounters (also used by the confusion
        # matrix); "min_dist" is the closest center-to-center distance to any
        # collision entity while the encounter was active. Records are finalized
        # when tracking ends, or on reset / terminal step mid-encounter.
        # "resolved": ended by the classifier's tracked -> None transition (TS out
        # of forget range or despawned) and the OS passed the TS; cut-off
        # encounters are not resolved. min_dist and CRI statistics only use
        # "valid" encounters (resolved and not collided).
        self._in_encounter: bool = False
        self._current_encounter_min_dist: float = float("inf")
        self._current_encounter_min_dist_bank: float = float("inf")
        self._current_encounter_collided: bool = False
        self._current_encounter_passed: bool = False
        self._current_encounter_max_cri: float = 0.0
        self._current_encounter_min_cri: float = float("inf")
        self._current_encounter_cri_sum: float = 0.0
        self._current_encounter_cri_count: int = 0
        # Time to first action: steps from encounter start to the first
        # |rudder| above the threshold. CTE is only accumulated during an
        # encounter so path-following segments do not dilute it.
        self._first_action_rudder_threshold_rad: float = math.radians(10.0)
        self._current_encounter_start_step: Optional[int] = None
        self._current_encounter_first_action_step: Optional[int] = None
        self._current_encounter_cte_sum: float = 0.0
        self._current_encounter_cte_count: int = 0
        self._current_encounter_max_cte: float = 0.0

        self._last_collision_info = {}
        self._colreg_violations = 0
        self._colreg_correct = 0
        self.ts_passed = None
        self._prev_passing_signal = None
        self._prev_passing_ts_idx = None
        self._prev_os_pos = None
        self._prev_ts_pos = None

        # Classification metrics (GT vs geometric), one record per encounter
        # (TS tracked until tracked_ts returns to None or the episode ends).
        # (gt, pred) is captured once at encounter entry, matching the label
        # locking used for reward shaping.
        self._episode_encounters: list[dict] = []    # encounters in the current episode
        self._all_encounters: deque = deque(maxlen=self.METRIC_HISTORY_MAXLEN)  # across episodes (capped, see above)
        self._prev_tracked_ts: Optional[int] = None  # for None -> tracked transition detection

        # OS/TS positions at the moment of passing.
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

        self._seg_starts: Optional[np.ndarray] = None
        self._seg_tangents: Optional[np.ndarray] = None
        self._seg_lengths: Optional[np.ndarray] = None

        # Episode step counter (reset per episode by _initialize_simulation);
        # unlike step_count it is not reset at checkpoints.
        self._episode_step: int = 0

        # Path segment around the OS, captured in step() before the reward may
        # advance the checkpoint; used for the collision side / CTE.
        self._collision_segment_start: np.ndarray = np.zeros(2, dtype=np.float32)
        self._collision_segment_end: np.ndarray = np.zeros(2, dtype=np.float32)

        # Render state (reset per episode in env_specific_reset, artists created in render()).
        self._passing_snapshot_artists: list = []
        self.os_trail_history: list = []
        self.ts_trail_history: list = []
        self.ts_trails: list = []

        super().__init__(
            render_mode=render_mode,
            time_step=time_step,
            max_steps=max_steps,
            verbose=verbose,
            start_pos=ship_pos,
            target_pos=target_pos,
            overall_contour=overall_contour,
            depth_contour=depth_contour,
            trajectory=trajectory,
            dynamics_model=dynamics_model,
            ship_model=ship_model,
            wind=wind,
            current=current,
            minimal_path_render=minimal_path_render,
            ugent_surrogate_config=ugent_surrogate_config,
            ugent_b01_surrogate_config=ugent_b01_surrogate_config,
        )

        self.scenario_builder = ScenarioBuilder(self)
        self.scenario_builder.cache_checkpoint_segments()
        self.forget_range = self.encounter_range + 20

        self.scenario_classifier = ScenarioClassifier(
            classify_range=self.encounter_range,
            forget_range=self.forget_range,
        )
        # Path segments let classify_path_projected() project the OS onto its segment.
        self.scenario_classifier.set_path_segments(
            self._seg_starts, self._seg_tangents, self._seg_lengths,
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

        # Initial scenario sampling. on_reset() also initializes the builder's
        # per-episode state in step-level mode; it only builds entities at
        # episode level.
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
    def _compute_cri(self, os_pos, os_heading, os_u, os_v, ts) -> Tuple[float, float, float]:
        """
        (CRI, TCPA [s], DCPA [m]) for a target ship; TCPA > 0 when approaching.
        Returns (0, 0, current range) when the relative velocity is ~0.
        """

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
            return 0.0, 0.0, math.sqrt(dx * dx + dy * dy)

        tcpa = -(dx * dvx + dy * dvy) / vr_sq
        dcpa_x = dx + dvx * tcpa
        dcpa_y = dy + dvy * tcpa
        dcpa = math.sqrt(dcpa_x * dcpa_x + dcpa_y * dcpa_y)

        cri_dcpa = self.cri_k1 * math.exp(self.cri_alpha1 * dcpa / self.cri_D)

        if tcpa <= 0.0:
            cri_tcpa = 0.0
        else:
            cri_tcpa = self.cri_k2 * math.exp(self.cri_alpha2 * tcpa / self.cri_T)

        cri = self.cri_omega1 * cri_dcpa + self.cri_omega2 * cri_tcpa
        return cri, tcpa, dcpa

    # -----------------------
    # Dynamics update & step
    # -----------------------
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one environment step, including collision entities and encounter metrics.
        """

        action = np.asarray(action, dtype=np.float32)

        if not self.phys_sim.state.terminated and not self.phys_sim.state.truncated:
            self.previous_action = self.performed_action.copy()
            self.previous_ship_pos = self.ship_pos.copy()
            self.previous_heading = self.phys_sim.state.heading

            self.phys_sim.step(action, self.enable_smoothing, self.enable_rudder_angle_scaling)
            self.ship_pos = np.array(
                [self.phys_sim.state.x, self.phys_sim.state.y], dtype=np.float32
            )
            self.performed_action = self.phys_sim.actuators.current_actuator_state.copy()
            self.step_count += 1
            self._episode_step += 1

        # Capture the collision segment before the reward may advance the checkpoint.
        prev_idx = max(0, self.checkpoint_index - 1)
        self._collision_segment_start = np.asarray(self.checkpoints[prev_idx]['pos'], dtype=np.float32)
        curr_idx = min(self.checkpoint_index, len(self.checkpoints) - 1)
        self._collision_segment_end = np.asarray(self.checkpoints[curr_idx]['pos'], dtype=np.float32)

        self._step_collision_entities()

        if self.scenario_obstacle_level == "step":
            self.scenario_builder.on_step(self.time_step)

        # Minimum distance to collision entities.
        if getattr(self, "collision_entities", None):
            for entity in self.collision_entities:
                dist = entity.distance_to_point(self.ship_pos)
                if dist < self._current_episode_min_dist:
                    self._current_episode_min_dist = dist
                if self._in_encounter and dist < self._current_encounter_min_dist:
                    self._current_encounter_min_dist = dist

        # Minimum distance to the bank within an encounter.
        if self._in_encounter and getattr(self, "polygon_shape", None) is not None:
            d_bank = float(self.polygon_shape.boundary.distance(Point(self.ship_pos)))
            if d_bank < self._current_encounter_min_dist_bank:
                self._current_encounter_min_dist_bank = d_bank

        # Sample CRI for the tracked TS during an active encounter.
        if self._in_encounter:
            ts_idx = self.scenario_classifier.tracked_ts
            if ts_idx is not None and ts_idx < len(self.collision_entities):
                ts = self.collision_entities[ts_idx]
                cri_val, _, _ = self._compute_cri(
                    self.ship_pos,
                    self.phys_sim.state.heading,
                    float(self.phys_sim.state.u),
                    float(self.phys_sim.state.v),
                    ts,
                )
                if cri_val > self._current_encounter_max_cri:
                    self._current_encounter_max_cri = cri_val
                if cri_val < self._current_encounter_min_cri:
                    self._current_encounter_min_cri = cri_val
                self._current_encounter_cri_sum += cri_val
                self._current_encounter_cri_count += 1

            # One CTE sample per encounter step; cross_error comes from the previous
            # step's reward call (one-step lag, like min_dist and CRI here).
            cte_val = float(getattr(self, "cross_error", 0.0))
            if not (math.isnan(cte_val) or math.isinf(cte_val)):
                abs_cte = abs(cte_val)
                self._current_encounter_cte_sum += abs_cte
                self._current_encounter_cte_count += 1
                if abs_cte > self._current_encounter_max_cte:
                    self._current_encounter_max_cte = abs_cte

            # First significant rudder action since encounter start, on the
            # physical rudder angle [rad].
            if (
                self._current_encounter_first_action_step is None
                and self.performed_action is not None
            ):
                # performed_action[0] is already the rudder angle in rad; do not
                # scale it by max_rudder_angle.
                abs_rudder_rad = abs(float(self.performed_action[0]))
                if abs_rudder_rad >= self._first_action_rudder_threshold_rad:
                    self._current_encounter_first_action_step = self._episode_step

        self._on_entities_stepped()

        # Reward and termination. After the episode has ended the stored flags
        # are returned again, so a finished episode never "un-terminates".
        reward = 0.0
        terminated = bool(self.phys_sim.state.terminated)
        truncated = bool(self.phys_sim.state.truncated)

        if self.phys_sim.state.terminated and self.phys_sim.state.truncated:
            logger.debug("Both terminated and truncated set simultaneously.")

        if not self.phys_sim.state.terminated and not self.phys_sim.state.truncated:
            reward, terminated = self._calculate_colav_reward()

            # Collision detection (shared by all subclasses).
            collision_entity_collision = False
            if getattr(self, "collision_entities", None):
                collision_entity_collision = self.scenario_builder.check_collision_entity_collision()
                if collision_entity_collision:
                    terminated = True

            bank_collision = not self.ship_in_open_water(self.ship_pos)
            if bank_collision:
                terminated = True

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

            success = self.checkpoint_index >= len(self.checkpoints)
            if terminated and not success:
                reward = -250.0

            if self.step_count >= self.max_steps and not terminated:
                # reward = -250
                truncated = True

            # Focused-encounter training: the builder requests truncation when the
            # post-encounter countdown elapses (no failure penalty).
            if (
                not terminated
                and not truncated
                and getattr(self.scenario_builder, "_truncate_requested", False)
            ):
                truncated = True

            self.phys_sim.state.terminated = terminated
            self.phys_sim.state.truncated = truncated

        # Collided episodes/encounters are excluded from min-distance statistics.
        if self._last_collision_info.get("collision"):
            self._current_episode_collided = True
            if self._in_encounter:
                self._current_encounter_collided = True

        obs = self._get_obs()

        # After _get_obs() has re-run the classifier for this step.
        self._update_encounter_metrics()

        info = {
            "rewards": reward,
            "terminations": terminated,
            "truncations": truncated,
            "collision_info": self._last_collision_info,
            "colreg_violations": self._colreg_violations,
            "colreg_correct": self._colreg_correct,
            "min_distance": self._current_episode_min_dist,
            "classification": {
                "encounters": list(self._episode_encounters),
            },
        }

        # Gymnasium contract: max-steps and focused-encounter cut-offs are
        # truncated (bootstrap V(s')); collisions and success are terminated.
        return obs, float(reward), bool(terminated), bool(truncated), info

    # --------------------------------
    # Reward (override in subclass)
    # --------------------------------
    def _calculate_colav_reward(self) -> Tuple[float, bool]:
        """
        COLAV reward; subclasses override with their own shaping. Collision
        detection and the failure penalty are handled by step().

        Returns:
            (reward, terminated)
        """

        return super()._calculate_reward()

    # -----------------------
    # Hooks for subclasses
    # -----------------------
    def _on_entities_stepped(self) -> None:
        """
        Hook after the collision entities are stepped (e.g. sensor geometry updates).
        """

        pass

    def _on_episode_reset(self) -> None:
        """
        Hook at the end of env_specific_reset(), after scenario sampling.
        """

        pass

    # ---------------------------
    # Classification metrics
    # ---------------------------
    def _update_encounter_metrics(self) -> None:
        """
        Drive the per-encounter lifecycle from tracked_ts transitions (called
        from step() after _get_obs() re-ran the classifier):
          * terminated/truncated: finalize an active encounter (not resolved).
          * None to tracked: start an encounter, capture (gt, pred) and reset
            the per-encounter accumulators.
          * tracked to None: finalize the encounter (tracked_lost).
        """

        # On terminal steps the classifier is not re-run (stale state), but an
        # active encounter is still finalized, as not resolved.
        if self.phys_sim.state.terminated or self.phys_sim.state.truncated:
            self._finalize_active_encounter(resolved_normally=False)
            return

        clf = self.scenario_classifier
        curr = clf.tracked_ts
        prev = self._prev_tracked_ts
        self._prev_tracked_ts = curr

        # Encounter end (TS out of forget range or despawned); resolved only if
        # the OS also passed the TS.
        if prev is not None and curr is None:
            self._finalize_active_encounter(resolved_normally=True)
            return

        # Otherwise only act on encounter start.
        if curr is None or prev is not None:
            return

        gt_ext = clf.ground_truth_extended_label
        pred_ext = clf.geometric_extended_label if gt_ext is not None else None

        # Reset per-encounter accumulators; accumulation starts next step.
        self._in_encounter = True
        self._current_encounter_min_dist = float("inf")
        self._current_encounter_min_dist_bank = float("inf")
        self._current_encounter_collided = False
        self._current_encounter_passed = False
        self._current_encounter_max_cri = 0.0
        self._current_encounter_min_cri = float("inf")
        self._current_encounter_cri_sum = 0.0
        self._current_encounter_cri_count = 0
        # Uses the monotonic _episode_step, since step_count resets at every
        # checkpoint; time_to_first_action = (first_action - start) * time_step.
        self._current_encounter_start_step = self._episode_step
        self._current_encounter_first_action_step = None
        self._current_encounter_cte_sum = 0.0
        self._current_encounter_cte_count = 0
        self._current_encounter_max_cte = 0.0

        # Always record the start so n_encounters is correct. gt_ext is None for
        # an entity the builder did not tag (should not happen); such records
        # count as encounters but not for classification accuracy.
        self._episode_encounters.append({
            "gt": gt_ext,
            "pred": pred_ext,
            "min_dist": float("inf"),
            "min_dist_bank": float("inf"),
            "collided": False,
            "passed": False,
            "tracked_lost": False,
            "resolved": False,
            "max_cri": 0.0,
            "min_cri": 0.0,
            "mean_cri": 0.0,
            "mean_cte": 0.0,
            "max_cte": 0.0,
            "time_to_first_action": None,
        })

    def _finalize_active_encounter(self, resolved_normally: bool = False) -> None:
        """
        Write the per-encounter aggregates into the latest record and reset them.

        Args:
            resolved_normally: True when triggered by the classifier's tracked
                to None transition, False when cut off by termination,
                truncation or reset. The record is "resolved" only if this is
                True and the OS passed the TS.
        """

        if not self._in_encounter or not self._episode_encounters:
            self._in_encounter = False
            return
        rec = self._episode_encounters[-1]
        rec["min_dist"] = float(self._current_encounter_min_dist)
        rec["min_dist_bank"] = float(self._current_encounter_min_dist_bank)
        rec["collided"] = bool(self._current_encounter_collided)
        rec["passed"] = bool(self._current_encounter_passed)
        rec["tracked_lost"] = bool(resolved_normally)
        rec["resolved"] = bool(resolved_normally and self._current_encounter_passed)
        rec["max_cri"] = float(self._current_encounter_max_cri)
        # Without CRI samples min_cri would be inf; use the max instead.
        if self._current_encounter_cri_count > 0:
            rec["min_cri"] = float(self._current_encounter_min_cri)
        else:
            rec["min_cri"] = float(self._current_encounter_max_cri)
        rec["mean_cri"] = (
            float(self._current_encounter_cri_sum / self._current_encounter_cri_count)
            if self._current_encounter_cri_count > 0 else 0.0
        )
        rec["mean_cte"] = (
            float(self._current_encounter_cte_sum / self._current_encounter_cte_count)
            if self._current_encounter_cte_count > 0 else 0.0
        )
        rec["max_cte"] = float(self._current_encounter_max_cte)
        # Time to first significant rudder action [s]; None if none occurred.
        if (
            self._current_encounter_first_action_step is not None
            and self._current_encounter_start_step is not None
        ):
            dt_steps = (
                self._current_encounter_first_action_step
                - self._current_encounter_start_step
            )
            rec["time_to_first_action"] = float(max(0, dt_steps) * self.time_step)
        else:
            rec["time_to_first_action"] = None
        self._in_encounter = False
        self._current_encounter_min_dist = float("inf")
        self._current_encounter_min_dist_bank = float("inf")
        self._current_encounter_collided = False
        self._current_encounter_passed = False
        self._current_encounter_max_cri = 0.0
        self._current_encounter_min_cri = float("inf")
        self._current_encounter_cri_sum = 0.0
        self._current_encounter_cri_count = 0
        self._current_encounter_start_step = None
        self._current_encounter_first_action_step = None
        self._current_encounter_cte_sum = 0.0
        self._current_encounter_cte_count = 0
        self._current_encounter_max_cte = 0.0
        # Clear the pass-once guard: in step-level mode each new TS is at index
        # 0, so a stale ts_passed would block every later pass detection.
        self.ts_passed = None

    # ---------------------------
    # Collision-entity helpers
    # ---------------------------
    def _sync_collision_entity_views(self) -> None:
        """
        Update the cached entity positions, radii and count used by observations and rendering.
        """

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
        """
        Advance the dynamic collision entities one step (activation, encounter behavior, path following).
        """

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

            # Dormant activation (single-encounter eval mode) for entities tagged
            # by the builder (overtaking / head-on); crossings use their own
            # activation below.
            if getattr(entity, "dormant_pending_activation", False) and active_ship_positions:
                entity_pos = entity.get_position()
                min_dist_os = min(
                    float(np.linalg.norm(entity_pos - os_pos))
                    for os_pos in active_ship_positions
                )
                act_range = float(getattr(entity, "dormant_activation_range", self.encounter_range))
                if min_dist_os <= act_range:
                    orig_speed = float(getattr(entity, "_dormant_original_speed", 0.0))
                    entity.dormant_pending_activation = False
                    entity.set_speed(orig_speed)
                    if hasattr(entity, "path_speed"):
                        entity.path_speed = orig_speed

            # Head-on dodge is disabled. To re-enable, for a head-on entity that has
            # not dodged yet, with min_dist = its distance to the nearest OS:
            # if self.enable_headon_dodge and min_dist <= self.encounter_range / 2:
            #     entity.dodged = True
            #     dodge_angle = math.radians(20)
            #     entity.set_heading(entity.get_heading() - dodge_angle)
            #     entity.set_lateral_speed(0.0)
            #     entity.route_mode = None

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
        """
        Move an entity along the checkpoint path, honoring its path direction
        and lateral (starboard) offset; marks route_finished at the path end.

        Args:
            entity: Collision entity to advance.
            dt: Simulation time step in seconds.
        """

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
        """
        Reset episode-specific state and build the collision entities.
        """

        # Finalize the previous episode's min distance (collided episodes excluded).
        if hasattr(self, "_current_episode_min_dist"):
            if not self._current_episode_collided and self._current_episode_min_dist < float("inf"):
                self._episode_min_distances.append(self._current_episode_min_dist)
        self._current_episode_min_dist = float("inf")
        self._current_episode_collided = False

        # Defensive: an encounter still active here was cut off, so not resolved.
        if hasattr(self, "_finalize_active_encounter"):
            self._finalize_active_encounter(resolved_normally=False)

        self.collision_entities = []
        self.obstacle_count = 0
        self.obstacle_pos = None
        self.static_ship_radii = []

        self._last_collision_info = {}

        self.scenario_classifier.reset()
        self.ts_passed = None
        self._colreg_violations = 0
        self._colreg_correct = 0
        self._prev_passing_signal = None
        self._prev_passing_ts_idx = None

        # Move the previous episode's encounters to the global accumulator
        # (used by plot_scenario_confusion_matrix).
        if self._episode_encounters:
            self._all_encounters.extend(self._episode_encounters)
        self._episode_encounters = []
        self._prev_tracked_ts = None

        for artists in self._passing_snapshot_artists:
            for artist in artists:
                artist.remove()
        self._passing_snapshot_artists = []
        self._passing_snapshots = []

        # Always call on_reset() so per-episode builder state (e.g. step_counter)
        # is also reset in step-level mode.
        self.scenario_builder.on_reset()

        self.os_trail_history = []
        for trail_line in self.ts_trails:
            trail_line.remove()
        self.ts_trail_history = []
        self.ts_trails = []

        self._on_episode_reset()

    # -----------------------
    # Rendering
    # -----------------------
    def _initialize_rendering(self) -> None:
        """
        Create the artists for own ship, target-ship trails and passing annotations.
        """

        super()._initialize_rendering()

        self.ship_plot_colav, = self.ax.plot([], [], 'bo', markersize=10, label='OS')
        self.heading_line_colav, = self.ax.plot([], [], color='black', linewidth=2)

        self.os_trail_history = []
        self.os_trail_line, = self.ax.plot(
            [], [], color='blue', alpha=0.5, linewidth=1.5, linestyle='-', label='OS trail',
        )

        # TS trail lines are created dynamically.
        self.ts_trails = []
        self.ts_trail_history = []

        self._passing_snapshot_artists = []

        # Blitted artists must be animated, so a full redraw (e.g. when a TS
        # despawns) does not bake them into the cached background as ghosts.
        self.ship_plot_colav.set_animated(True)
        self.heading_line_colav.set_animated(True)
        self.os_trail_line.set_animated(True)

    def _get_collision_entity_render_key(self):
        """
        Hashable key that changes whenever the rendered collision entities change.
        """

        entities = getattr(self, "collision_entities", []) or []
        if not entities:
            return "none", 0
        rounded = tuple(
            (float(round(entity.x, 2)), float(round(entity.y, 2)), float(round(entity.radius, 2)), entity.entity_type)
            for entity in entities
        )
        return len(entities), rounded

    def _clear_collision_entity_patches(self) -> None:
        """
        Remove all collision-entity patches from the plot.
        """

        if not hasattr(self, "_collision_entity_patches"):
            self._collision_entity_patches = []
            return
        for patch in self._collision_entity_patches:
            # Best effort: a patch that cannot be removed is skipped.
            # noinspection PyBroadException
            try:
                patch.remove()
            except Exception:
                pass
        self._collision_entity_patches = []

    def _draw_collision_entities(self) -> None:
        """
        Draw the collision-entity safety circles.
        """

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
        """
        Draw static landmarks (start position = the episode's initial ship position).
        """

        self.start_pos = np.asarray(self.initial_ship_pos, dtype=np.float32).copy()
        super()._draw_fixed_landmarks()

    def render(self) -> None:
        """
        Render (human mode only) with collision entities, trails and passing snapshots.
        """

        if self.render_mode != 'human':
            return

        if self.initialize_plots and not hasattr(self, 'ship_plot_colav'):
            self._initialize_rendering()
            self._draw_fixed_landmarks()
            self._draw_collision_entities()
            plt.show(block=False)
            plt.pause(0.001)
            self.fig.canvas.draw()
            self.background = self.fig.canvas.copy_from_bbox(self.ax.bbox)
            self.fig.canvas.mpl_connect('draw_event', self._on_draw)

        canvas = self.fig.canvas

        def _safe(callable_obj):
            """
            Call callable_obj(), ignoring rendering errors so a flaky draw does not abort the run.
            """

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
            trail_line.set_animated(True)
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

        self._render_subclass(heading)

        # Passing snapshot artists.
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
                mid_x, mid_y, f"{dist:.0f}m", fontsize=12, color='gray',
                fontweight='bold', ha='center', va='bottom', zorder=6,
            )
            kind = snap.get("kind", "passed")
            tag = "Start" if kind == "start" else "Passed"
            # Labels below the markers, offset in points (independent of axis scale).
            os_label = self.ax.annotate(
                f"{tag}", xy=(os_p[0], os_p[1]),
                xytext=(0, -10), textcoords='offset points',
                fontsize=12, color='blue',
                ha='center', va='top', zorder=6,
            )
            ts_label = self.ax.annotate(
                f"{tag}", xy=(ts_p[0], ts_p[1]),
                xytext=(0, -10), textcoords='offset points',
                fontsize=12, color='red',
                ha='center', va='top', zorder=6,
            )
            for artist in (os_marker, ts_marker, dist_line, dist_text, os_label, ts_label):
                artist.set_animated(True)
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
        _safe(lambda: self.fig.canvas.start_event_loop(0.0001))

    def _render_subclass(self, heading: float) -> None:
        """
        Hook for subclass rendering (e.g. polar beams).
        """

        pass

    def _blit_subclass(self) -> None:
        """
        Hook to blit additional subclass artists (e.g. polar beams).
        """

        pass
