from datetime import time

from fw.environments.py_sim_colav_base import CHECKPOINTS_DISTANCE, PySimColavBaseEnv
from fw.environments.py_sim_env import (
    MAX_CHECKPOINTS_DISTANCE,
    calculate_perpendicular_lines,
    CHECKPOINT_AREA_SIZE,
    TARGET_AREA_SIZE,
    SUCCESS_REWARD,
    PERPENDICULAR_LINE_LENGTH,
)
from fw.re_planner.collision_detection import CollisionDetector
from fw.re_planner.RRT import RRTStarReplanner
from fw.re_planner.APF import ArtificialPotentialField
from fw.re_planner.RRT_APF import RRTAPFReplanner

from typing import Optional, Tuple, Dict, List
from fw.ships.ship_factory import ShipModel
from fw.dynamics.dynamics_model import Model
import numpy as np
import time



class PySimColavPlannerEnv(PySimColavBaseEnv):
    """Collision avoidance environment using the detection-radius + RRT replanner approach.

    This class keeps the new scenario/entity management from PySimColavBaseEnv,
    and restores the old replanner behaviour:
    - detect target ships inside a detection radius
    - predict future collisions
    - trigger RRT replanning when needed
    """

    def __init__(
            self,
            render_mode: Optional[str] = None,
            time_step: float = 2,
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
            encounter_range: float = 500.0,
            eval: bool = True,
            # replanner parameters
            danger_distance: float = 50.0,
            time_horizon: float = 500.0,
            ships_inside_observation: int = 3,
            detection_radius: float = 2000.0,
            rrt_replan_cooldown_steps: int = 0,
            # Scenario configuration (passed to base)
            scenario_types: Optional[list] = ["crossing-right", "head-on", "overtaking", "static"],
            scenario_probs: Optional[list] = [0.25, 0.25, 0.25, 0.25],
            obstacle_population: Optional[list] = [1],
            obstacle_weights: Optional[list] = [1],
            scenario_obstacle_level: Optional[str] = "episode",
    ) -> None:

        self.ships_inside_observation = int(ships_inside_observation)
        self.detection_radius = float(detection_radius)
        self.collision_detector = CollisionDetector(float(danger_distance), float(time_horizon))
        self.replanner = RRTStarReplanner(danger_radius=float(danger_distance), full_circle=False)
        self.rrt_replan_cooldown_steps = int(rrt_replan_cooldown_steps)
        self._last_rrt_replan_step = -10 ** 9
        self._last_rrt_replan_info: Dict = {}

        # Keep planner env aligned with the old behaviour: adaptive scenarios at step level.
        scenario_obstacle_level = str(scenario_obstacle_level) if scenario_obstacle_level is not None else "step"

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
            encounter_range=encounter_range,
            eval=eval,
            scenario_types=scenario_types,
            scenario_probs=scenario_probs,
            obstacle_population=obstacle_population,
            obstacle_weights=obstacle_weights,
            scenario_obstacle_level=scenario_obstacle_level,
        )

        self.original_path = np.asarray(
            [checkpoint["pos"] for checkpoint in self.checkpoints],
            dtype=np.float32,
        ).copy()
        self.draw_rrt_paths = True

        # Override base lateral max specifically for this planner subclass
        self.static_ship_lateral_max = 0.0

        # Save the closest point of approach for visualization
        self.cpa_points = []

    def _calculate_colav_reward(self) -> Tuple[float, bool]:
        """Replanner-specific reward.

        Reward stays identical to the old replanner version: path-following
        reward from the parent environment. Collision handling remains in
        PySimColavBaseEnv.step().
        """
        return super()._calculate_colav_reward()

    def _on_episode_reset(self) -> None:
        self._last_rrt_replan_info = {"triggered": False, "success": False}
        self._last_rrt_replan_step = -10 ** 9

    def _get_obs(self) -> np.ndarray:
        """Restore the old detection-radius + collision-prediction + RRT trigger flow."""

        self._last_rrt_replan_info = {"triggered": False, "success": False}

        entities = getattr(self, "collision_entities", []) or []
        detected: List = []
        for entity in entities:
            dist_to_ship = float(entity.distance_to_point(self.ship_pos))
            if dist_to_ship <= self.detection_radius:
                detected.append((dist_to_ship, entity))

        detected.sort(key=lambda item: item[0])
        detected = detected[:self.ships_inside_observation]

        if detected:
            #print(f"Detected {len(detected)} ship(s) inside detection range.")
            obstacles_entities = [entity for _, entity in detected]
            remaining_path = np.asarray(
                [cp["pos"] for cp in self.checkpoints[max(0, int(self.checkpoint_index) - 1):]],
                dtype=np.float32,
            )

            velocities = np.array(
                [self.phys_sim.state.u, self.phys_sim.state.v],
                dtype=np.float32,
            )

            collision_detected, cpa_point, tcpa, dcpa, relative_heading = self.collision_detector.predict_collision(
                obstacles_entities,
                self.ship_pos,
                remaining_path,
                velocities,
            )
            if any(collision_detected):
                self.cpa_points.append(cpa_point)
                #print(f"Collision predicted with {len(obstacles_entities)} obstacle(s). Triggering replanner. Details: entities={obstacles_entities}, collision_detected={collision_detected}, cpa_point={cpa_point}, tcpa={tcpa}, dcpa={dcpa}, relative_heading={relative_heading}")
                self._maybe_replan_rrt(
                    obstacles_entities=obstacles_entities,
                    collision_detected=collision_detected,
                    cpa_point=cpa_point,
                    tcpa=tcpa,
                    relative_heading=relative_heading,
                )
        obs = super()._get_obs()


        # Add two additional future checkpoints to the observation.
        # This assumes _get_obs() is still the original version, without these 6 values.
        current_chkp_idx = self.checkpoint_index
        next_chkp_idx = current_chkp_idx + 1
        next2_chkp_idx = current_chkp_idx + 2
        next3_chkp_idx = current_chkp_idx + 3

        # Get checkpoint positions, extrapolating beyond the final checkpoint if needed
        if next_chkp_idx <= len(self.checkpoints) - 1:
            next_chkp_pos = self.checkpoints[next_chkp_idx]['pos']
        else:
            multiplier = next_chkp_idx - (len(self.checkpoints) - 1)
            next_chkp_pos = (
                self.checkpoints[len(self.checkpoints) - 1]['pos']
                + 2 * multiplier * self.direction_last_path
            )

        if next2_chkp_idx <= len(self.checkpoints) - 1:
            next2_chkp_pos = self.checkpoints[next2_chkp_idx]['pos']
        else:
            multiplier = next2_chkp_idx - (len(self.checkpoints) - 1)
            next2_chkp_pos = (
                self.checkpoints[len(self.checkpoints) - 1]['pos']
                + 2 * multiplier * self.direction_last_path
            )

        if next3_chkp_idx <= len(self.checkpoints) - 1:
            next3_chkp_pos = self.checkpoints[next3_chkp_idx]['pos']
        else:
            multiplier = next3_chkp_idx - (len(self.checkpoints) - 1)
            next3_chkp_pos = (
                self.checkpoints[len(self.checkpoints) - 1]['pos']
                + 2 * multiplier * self.direction_last_path
            )

        # Distance to checkpoint +2
        norm_next2_distance = self._logarithmic_scaling(
            value=float(np.linalg.norm(self.ship_pos - next2_chkp_pos)),
            value_range=(0, 3 * MAX_CHECKPOINTS_DISTANCE),
            half_value=2.5 * CHECKPOINTS_DISTANCE
        )

        # Distance to checkpoint +3
        norm_next3_distance = self._logarithmic_scaling(
            value=float(np.linalg.norm(self.ship_pos - next3_chkp_pos)),
            value_range=(0, 4 * MAX_CHECKPOINTS_DISTANCE),
            half_value=3.5 * CHECKPOINTS_DISTANCE
        )

        # Heading error to checkpoint +2
        heading_error3 = self._calculate_heading_error(
            self._safe_heading_from_vector(next2_chkp_pos - self.ship_pos),
            self.phys_sim.state.heading,
            0.0
        )

        # Parallel heading error for segment next -> next2
        heading_error_parallel3 = self._calculate_heading_error(
            self._safe_heading_from_vector(next2_chkp_pos - next_chkp_pos),
            self.phys_sim.state.heading,
            0.0
        )

        # Heading error to checkpoint +3
        heading_error4 = self._calculate_heading_error(
            self._safe_heading_from_vector(next3_chkp_pos - self.ship_pos),
            self.phys_sim.state.heading,
            0.0
        )

        # Parallel heading error for segment next2 -> next3
        heading_error_parallel4 = self._calculate_heading_error(
            self._safe_heading_from_vector(next3_chkp_pos - next2_chkp_pos),
            self.phys_sim.state.heading,
            0.0
        )

        obs = np.hstack([
            obs,

            [2 * norm_next2_distance - 1],          # Distance to checkpoint +2
            [heading_error3 / np.pi],               # Heading error to checkpoint +2
            [heading_error_parallel3 / np.pi],      # Parallel heading error, next -> next2

            [2 * norm_next3_distance - 1],          # Distance to checkpoint +3
            [heading_error4 / np.pi],               # Heading error to checkpoint +3
            [heading_error_parallel4 / np.pi],      # Parallel heading error, next2 -> next3
        ]).astype(np.float32)

        return obs

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]: #NON USA LE FUNZIONI DI QUESTO AMBIENTE
        """Keep base stepping logic and expose replanning diagnostics like the old env."""

        obs, reward, terminated, truncated, info = super().step(action)
        info["rrt_replan_info"] = dict(self._last_rrt_replan_info)
        return obs, reward, terminated, truncated, info

    def _rebuild_path_checkpoints(self, new_path: np.ndarray) -> bool:
        """Replace the active route with a new polyline and regenerate checkpoints."""
        pts = np.asarray(new_path, dtype=np.float32)
        if pts.ndim != 2 or pts.shape[1] != 2 or len(pts) < 2:
            return False

        dedup = [pts[0]]
        for p in pts[1:]:
            if np.linalg.norm(p - dedup[-1]) > 1e-6:
                dedup.append(p)
        if len(dedup) < 2:
            return False

        path = np.asarray(dedup, dtype=np.float32)
        checkpoints = []
        n = len(path)
        for idx, point in enumerate(path):
            checkpoints.append(
                {
                    "pos": point.copy(),
                    "radius": float(CHECKPOINT_AREA_SIZE),
                    "reward": (idx / max(1, n - 1)) * 10.0,
                }
            )

        checkpoints[-1]["radius"] = float(TARGET_AREA_SIZE)
        checkpoints[-1]["reward"] = float(SUCCESS_REWARD)

        lines = calculate_perpendicular_lines(checkpoints, line_length=PERPENDICULAR_LINE_LENGTH)
        for cp, line in zip(checkpoints, lines):
            cp["perpendicular_line"] = line

        self.checkpoints = checkpoints
        self.path = np.array([cp["pos"] for cp in checkpoints], dtype=np.float32)
        self.target_pos = self.checkpoints[-1]["pos"].copy()

        if hasattr(self, "scenario_builder"):
            self.scenario_builder.cache_checkpoint_segments()

        return True

    def _realign_checkpoint_indices_to_path(self) -> None:
        """Align the current checkpoint index with the next waypoint on the active path."""
        if self.path is None or len(self.path) < 2:
            return

        ship_pos = np.asarray(self.ship_pos, dtype=np.float32)
        d = np.linalg.norm(self.path - ship_pos, axis=1)
        nearest_idx = int(np.argmin(d))
        next_idx = min(max(nearest_idx + 1, 1), len(self.checkpoints) - 1)
        self.checkpoint_index = next_idx

    def _build_rrt_trajectory_from_current_ship_state(self) -> Optional[np.ndarray]:
        """Build the full checkpoint path used as input to the RRT replanner.

        Kept aligned with the old planner: the RRT receives the current active
        checkpoint polyline, not a newly constructed ship-position-prefixed path.
        """

        if not getattr(self, "checkpoints", None):
            return None

        remaining = [np.asarray(cp["pos"], dtype=np.float32) for cp in self.checkpoints]
        if len(remaining) < 2:
            return None

        dedup = [remaining[0]]
        for p in remaining[1:]:
            if float(np.linalg.norm(p - dedup[-1])) > 1e-6:
                dedup.append(p)

        if len(dedup) < 2:
            return None

        return np.asarray(dedup, dtype=np.float32)

    def _maybe_replan_rrt(
        self,
        obstacles_entities: List,
        collision_detected: List[bool],
        cpa_point: List,
        tcpa: List,
        relative_heading: List,
    ) -> None:
        """Trigger RRT replanning when a predicted collision is detected."""

        current_step = int(self.step_count)
        steps_since_last = current_step - int(self._last_rrt_replan_step)

        if steps_since_last < int(self.rrt_replan_cooldown_steps):
            self._last_rrt_replan_info = {
                "triggered": False,
                "success": False,
                "reason": "cooldown",
                "steps_until_next": int(self.rrt_replan_cooldown_steps - steps_since_last),
            }
            return

        collision_indices = [idx for idx, hit in enumerate(collision_detected) if bool(hit)]
        if not collision_indices:
            return

        def _tcpa_sort_key(idx: int) -> float:
            val = tcpa[idx]
            if val is None:
                return float("inf")
            return float(val) if float(val) >= 0.0 else float("inf")

        best_idx = min(collision_indices, key=_tcpa_sort_key)
        obstacle = obstacles_entities[best_idx]

        heading = relative_heading[best_idx]
        print(f"relative heading: {heading}")

        # Detector convention:
        #   0°/360°: head-on, 90°: crossing-right,
        #   180°: overtaking/static, 270°: crossing-left.
        if heading > 340 or heading <= 120:
            print("Head-on or crossing right collision detected. Triggering replanner with half circle.")
            self.replanner.full_circle = False
        elif heading > 120 and heading <= 240:
            print("Overtaking or static collision detected. Triggering replanner with full circle.")
            self.replanner.full_circle = True
        else:
            print("Crossing left collision detected. No triggering replanner.")
            return

        center_candidate = cpa_point[best_idx]
        if center_candidate is None:
            obstacle_center = np.asarray(obstacle.get_position(), dtype=np.float32)
        else:
            obstacle_center = np.asarray(center_candidate, dtype=np.float32)

        if obstacle_center.ndim != 1 or obstacle_center.shape[0] < 2:
            obstacle_center = np.asarray(obstacle.get_position(), dtype=np.float32)

        obstacle_center = obstacle_center[:2]
        obstacle_radius = float(getattr(obstacle, "radius", 0.0))

        trajectory = self._build_rrt_trajectory_from_current_ship_state()
        if trajectory is None:
            self._last_rrt_replan_info = {
                "triggered": True,
                "success": False,
                "reason": "invalid_trajectory",
            }
            self._last_rrt_replan_step = current_step
            return

        try:
            print("REPLANNING")
            start_time = time.time()
            
            replanned = self.replanner.replan(
                obstacle_center=obstacle_center,
                ship_trajectory=trajectory,
                obstacle_radius=obstacle_radius,
            )
            end_time = time.time()
            print(f"REPLANNING TOOK {end_time - start_time:.2f} seconds")
            print(replanned)
        except Exception as exc:
            print(f"replanning failed with an exception: {exc}")
            self._last_rrt_replan_info = {
                "triggered": True,
                "success": False,
                "reason": "planner_exception",
                "message": str(exc),
            }
            self._last_rrt_replan_step = current_step
            return

        success = bool(replanned.success) and self._rebuild_path_checkpoints(
            np.asarray(replanned.spliced_trajectory, dtype=np.float32)
        )

        if success:
            self._realign_checkpoint_indices_to_path()

        self._last_rrt_replan_info = {
            "triggered": True,
            "success": bool(success),
            "step": int(current_step),
            "collision_obstacle_index": int(best_idx),
            "iterations": int(replanned.iterations_used),
            "cost": float(replanned.cost),
            "planning_time_s": float(replanned.planning_time_s),
            "waypoints": int(len(replanned.spliced_trajectory)),
            "entry_idx": int(replanned.entry_idx),
            "exit_idx": int(replanned.exit_idx),
            "entry_wp": np.asarray(replanned.entry_wp, dtype=np.float32).tolist(),
            "exit_wp": np.asarray(replanned.exit_wp, dtype=np.float32).tolist(),
            "rrt_path": [np.asarray(point, dtype=np.float32).tolist() for point in replanned.rrt_path],
            "spliced_trajectory": [np.asarray(point, dtype=np.float32).tolist() for point in replanned.spliced_trajectory],
        }
        self._last_rrt_replan_step = current_step
