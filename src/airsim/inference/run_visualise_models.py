"""
Modular model visualisation runner for UE5.
"""

from __future__ import annotations

import sys
import math
import time
import logging
import argparse
import numpy as np

from pathlib import Path
from types import SimpleNamespace


_SRC_DIR = Path(__file__).resolve().parent.parent.parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from airsim.communication.airsim_bridge import AirSimBridge  # noqa: E402
from airsim.config.airsim_config import (  # noqa: E402
    AirSimConfig,
    DEFAULT_AIRSIM_SAVE_SLOT,
    DEFAULT_AIRSIM_SIMULATION_SPEED,
    DEFAULT_THROTTLE_VISUAL_GAIN,
    get_runtime_settings,
)
from airsim.inference.visualisation_models import (  # noqa: E402
    ModelVisualisationAdapter,
    get_supported_environment_names,
    get_supported_model_families,
    resolve_visualisation_adapter,
)
from fw.common.airsim.tools.airsim_export import (  # noqa: E402
    AirSimSessionExporter,
    build_framework_visualisation_row,
)
from fw.common.airsim.tools.trajectory_analysis import (  # noqa: E402
    build_run_summary_from_csv,
    format_performance_summary,
    render_side_by_side_trajectory_plot,
)
from fw.common.airsim.tools.path_utils import resolve_model_path  # noqa: E402
from fw.configuration.config import AIRSIM_COMPARISON_TRAJECTORY_PLOT_FILENAME  # noqa: E402

logger = logging.getLogger(__name__)

_DEFAULT_MIN_LOG_INTERVAL_S = 0.25
_DEFAULT_BRIDGE_TELEMETRY_HZ = 12.0
_DEFAULT_STARTUP_SETTLE_S = 0.4
_DEFAULT_TRAIL_MIN_SEGMENT_M = 1.5


def _normalise_policy_observation(obs, expected_dim: int | None = None) -> np.ndarray:
    """
    Flatten a framework observation (flat single-ship or batched per-ship) into
    a policy input vector.

    Args:
        obs: Observation returned by the framework environment.
        expected_dim: Optional policy input dimension, used to choose between
            flattening and taking the first ship row.

    Returns:
        np.ndarray: Flat observation vector for PPOPolicy.predict.
    """

    arr = np.asarray(obs, dtype=np.float32)

    if arr.ndim <= 1:
        return arr.reshape(-1)

    if arr.shape[0] == 1:
        return arr[0].reshape(-1)

    if expected_dim is not None and arr.shape[-1] == int(expected_dim):
        return arr[0].reshape(-1)

    flat = arr.reshape(-1)
    if expected_dim is not None and flat.size == int(expected_dim):
        return flat

    return arr[0].reshape(-1)


def _normalise_policy_action(action, expected_dim: int | None = None) -> np.ndarray:
    """
    Flatten a PPOPolicy.predict action (optional expected_dim from the action space).
    """

    arr = np.asarray(action, dtype=np.float32).reshape(-1)
    if expected_dim is None or arr.size <= int(expected_dim):
        return arr
    return arr[: int(expected_dim)]


class VisualiseModelsRunner:
    """
    Mirrors framework-physics model playback into UE5. Model-family specifics
    (policy loading, environment, state extraction) are delegated to a
    ModelVisualisationAdapter.
    """

    def __init__(self, adapter: ModelVisualisationAdapter, env_name: str = "auto") -> None:
        """
        Args:
            adapter: Adapter implementing family-specific replay behavior.
            env_name: Requested framework environment name, or "auto".
        """

        self.adapter = adapter
        self.env_name = env_name

    def _resolve_environment_name(self, model_path: str) -> str:
        """
        Canonical framework environment name for the replay of model_path.
        """

        return self.adapter.resolve_environment_name(model_path, self.env_name)

    def _load_policy(self, model_path: str, device: str):
        """
        Load the replay policy (an object exposing predict) on the given torch device.
        """

        return self.adapter.load_policy(model_path, device=device)

    def _build_environment(self, selected_env: str, max_steps: int):
        """
        Construct the framework environment selected_env with max_steps.
        """

        return self.adapter.build_environment(selected_env, max_steps=max_steps)

    def _motion_state_from_env(self, env, bridge: AirSimBridge):
        """
        Current motion state in the AirSim frame (via the adapter and the bridge's transforms).
        """

        return self.adapter.motion_state_from_env(env, bridge)

    def _animate_state(self, env, bridge: AirSimBridge, **kwargs):
        """
        Mirror one framework state transition into UE5; kwargs are animation parameters.
        """

        return self.adapter.animate_state(env, bridge, **kwargs)

    def _extract_pose(self, env) -> tuple[float, float, float]:
        """
        Current framework-frame pose (x, y, heading) via the adapter.
        """

        return self.adapter.extract_pose(env)

    def _extract_velocity(self, env) -> tuple[float, float, float]:
        """
        Current framework-frame velocity (surge, sway, yaw_rate) via the adapter.
        """

        return self.adapter.extract_velocity(env)

    def run(
        self,
        model_path: str,
        ip: str = "",
        port: int = 41451,
        max_steps: int = 14000,
        num_episodes: int = 1,
        device: str = "cpu",
        draw_calibration_reference: bool = True,
        log_every: int = 50,
        save_slot: str = DEFAULT_AIRSIM_SAVE_SLOT,
        simulation_speed: float = DEFAULT_AIRSIM_SIMULATION_SPEED,
        mirror_controls_in_ue5: bool = False,
        animate_steering_in_ue5: bool = True,
        animate_throttle_in_ue5: bool = True,
        render_interpolation_steps: int = 4,
        steering_visual_gain: float = 0.35,
        throttle_visual_gain: float = DEFAULT_THROTTLE_VISUAL_GAIN,
        use_kinematics_in_ue5: bool = False,
        min_render_hz: float = 90.0,
        max_translation_step: float = 0.15,
        max_yaw_step_deg: float = 1.5,
        lock_path_in_ue5: bool = False,
        vehicle_object_name: str = "",
        draw_ship_trail_in_ue5: bool = True,
        ship_trail_thickness: float = 14.0,
        ship_trail_min_segment_m: float = _DEFAULT_TRAIL_MIN_SEGMENT_M,
        min_log_interval_s: float = _DEFAULT_MIN_LOG_INTERVAL_S,
        bridge_telemetry_hz: float = _DEFAULT_BRIDGE_TELEMETRY_HZ,
        startup_settle_s: float = _DEFAULT_STARTUP_SETTLE_S,
        verbose: bool = False,
    ) -> None:
        """
        Run framework-model playback mirrored into UE5.

        Args:
            model_path: Path to the trained model checkpoint.
            ip: AirSim server IP address.
            port: AirSim RPC port.
            max_steps: Maximum number of framework steps per episode.
            num_episodes: Number of episodes to visualize.
            device: Inference device passed to the policy loader.
            draw_calibration_reference: Draw the calibration overlay in UE5.
            log_every: Log every N steps when logging is enabled.
            save_slot: AirSim save slot to load before playback.
            simulation_speed: UE5 simulation-speed multiplier.
            mirror_controls_in_ue5: Mirror framework controls into UE5.
            animate_steering_in_ue5: Animate steering while mirroring.
            animate_throttle_in_ue5: Animate throttle while mirroring.
            render_interpolation_steps: Number of interpolation steps per UE5 update.
            steering_visual_gain: Steering visualization gain.
            throttle_visual_gain: Throttle visualization gain.
            use_kinematics_in_ue5: Move the UE5 actor kinematically.
            min_render_hz: Minimum UE5 render update rate target.
            max_translation_step: Maximum translation distance per interpolation step.
            max_yaw_step_deg: Maximum yaw delta per interpolation step [deg].
            lock_path_in_ue5: Lock motion onto the framework path.
            vehicle_object_name: Optional UE5 vehicle object override.
            draw_ship_trail_in_ue5: Draw a blue trail behind the ship in UE5.
            ship_trail_thickness: Thickness of the ship trail.
            ship_trail_min_segment_m: Minimum movement before adding a new UE5 trail segment.
            min_log_interval_s: Minimum wall-clock interval between info logs.
            bridge_telemetry_hz: Maximum bridge telemetry polling rate.
            startup_settle_s: Startup settle window before stepping begins.
            verbose: Emit detailed per-step logging.
        """

        logger.setLevel(logging.INFO if verbose else logging.WARNING)

        selected_env = self._resolve_environment_name(model_path)
        resolved_model_path = resolve_model_path(model_path, _SRC_DIR)

        banner = "=" * 62
        print(banner)
        print("  MODEL VISUALISATION IN UE5")
        print(f"  Model      : {resolved_model_path.name}")
        print(f"  Family     : {self.adapter.family}")
        print(f"  Environment: {selected_env}")
        print(f"  Save slot  : {save_slot}")
        print(banner)

        policy = self._load_policy(model_path, device=device)
        with AirSimSessionExporter(
            session_name=f"visualisation_{self.adapter.family}_{resolved_model_path.stem}",
            metadata={
                "script": "airsim.inference.run_visualise_models",
                "model_path": str(resolved_model_path),
                "model_name": resolved_model_path.name,
                "model_family": self.adapter.family,
                "environment_name": selected_env,
                "model_origin": self.adapter.model_origin,
                "configuration": {
                    "ip": ip or "127.0.0.1",
                    "port": port,
                    "max_steps": max_steps,
                    "num_episodes": num_episodes,
                    "device": device,
                    "draw_calibration_reference": draw_calibration_reference,
                    "log_every": log_every,
                    "min_log_interval_s": float(min_log_interval_s),
                    "save_slot": save_slot,
                    "simulation_speed": simulation_speed,
                    "mirror_controls_in_ue5": mirror_controls_in_ue5,
                    "animate_steering_in_ue5": animate_steering_in_ue5,
                    "animate_throttle_in_ue5": animate_throttle_in_ue5,
                    "render_interpolation_steps": render_interpolation_steps,
                    "steering_visual_gain": steering_visual_gain,
                    "throttle_visual_gain": throttle_visual_gain,
                    "use_kinematics_in_ue5": use_kinematics_in_ue5,
                    "min_render_hz": min_render_hz,
                    "max_translation_step": max_translation_step,
                    "max_yaw_step_deg": max_yaw_step_deg,
                    "lock_path_in_ue5": lock_path_in_ue5,
                    "vehicle_object_name": vehicle_object_name,
                    "draw_ship_trail_in_ue5": draw_ship_trail_in_ue5,
                    "ship_trail_thickness": float(ship_trail_thickness),
                    "ship_trail_min_segment_m": float(ship_trail_min_segment_m),
                    "bridge_telemetry_hz": float(bridge_telemetry_hz),
                    "startup_settle_s": float(startup_settle_s),
                },
            },
            session_type="visualisation",
        ) as session_exporter:
            session_exporter.update_metadata(
                {
                    "resolved_model_path": str(resolved_model_path),
                    "export_directory": str(session_exporter.export_dir),
                    "bridge_telemetry_path": str(session_exporter.bridge_log_path),
                }
            )
            comparison_plot_path = session_exporter.export_dir / AIRSIM_COMPARISON_TRAJECTORY_PLOT_FILENAME
            bridge = None
            env = None

            def _trail_point_from_motion_state(motion_state: np.ndarray):
                """
                Vector3r-compatible trail point from an AirSim-frame [x, y, yaw, vx, vy, yaw_rate].
                """

                trail_offset_m = 8.0
                trail_x = float(motion_state[0] - np.cos(float(motion_state[2])) * trail_offset_m)
                trail_y = float(motion_state[1] - np.sin(float(motion_state[2])) * trail_offset_m)
                render_z = float(getattr(bridge, "render_z", 0.0))
                trail_z = float(render_z - 12.0)
                vector3r_factory = getattr(bridge.airsim, "Vector3r", None)
                if callable(vector3r_factory):
                    return vector3r_factory(trail_x, trail_y, trail_z)
                return SimpleNamespace(
                    x_val=trail_x,
                    y_val=trail_y,
                    z_val=trail_z,
                )

            try:
                bridge = AirSimBridge(
                    AirSimConfig(
                        ip=ip,
                        port=port,
                        csv_log_path=str(session_exporter.bridge_log_path),
                    )
                )
                if not bridge.is_connected:
                    raise ConnectionError(
                        f"AirSimBridge could not connect to UE5 at {ip or '127.0.0.1'}:{port}."
                    )

                env = self._build_environment(selected_env, max_steps=max_steps)
                max_yaw_step_rad = np.deg2rad(max_yaw_step_deg)
                bridge_telemetry_interval_s = 1.0 / max(float(bridge_telemetry_hz), 1.0)
                target_step_interval_s = env.time_step / max(float(simulation_speed), 1e-6)
                target_step_rate_hz = 1.0 / max(target_step_interval_s, 1e-6)
                steps_per_ue5_update = max(
                    1,
                    int(math.ceil(target_step_rate_hz / max(float(min_render_hz), 1.0))),
                )
                adaptive_settle_s = min(
                    max(float(startup_settle_s), 0.0),
                    max(0.05, target_step_interval_s * float(steps_per_ue5_update)),
                )
                last_bridge_telemetry_at = 0.0

                bridge.prepare_model_visualisation(
                    save_slot=save_slot,
                    simulation_speed=simulation_speed,
                    draw_reference=draw_calibration_reference,
                )

                if not mirror_controls_in_ue5 and not animate_steering_in_ue5 and not animate_throttle_in_ue5:
                    neutral_controls = bridge.airsim.CarControls()
                    neutral_controls.throttle = 0.0
                    neutral_controls.steering = 0.0
                    neutral_controls.brake = 0.0
                    bridge.set_controls(neutral_controls)

                for episode in range(num_episodes):
                    logger.info("-- Visualisation episode %d/%d --", episode + 1, num_episodes)
                    obs, _ = env.reset()
                    previous_trail_point = None
                    trail_points_history: list = []

                    # Flash the calibration overlay for 10 s at episode start for a
                    # visual check (episode 0: already drawn during setup).
                    if (
                        draw_calibration_reference
                        and episode > 0
                        and bridge.is_connected
                    ):
                        try:
                            bridge.redraw_calibration_reference(duration=3600.0)
                        except Exception as exc:
                            logger.debug("Episode-start overlay redraw failed: %s", exc)

                    previous_motion_state = self._motion_state_from_env(env, bridge)
                    previous_actuator_state = None
                    previous_motion_state, previous_actuator_state = self._animate_state(
                        env,
                        bridge,
                        target_motion_state=previous_motion_state,
                        previous_motion_state=None,
                        total_duration=0.0,
                        interpolation_steps=1,
                        mirror_controls_in_ue5=False,
                        animate_steering_in_ue5=animate_steering_in_ue5,
                        animate_throttle_in_ue5=animate_throttle_in_ue5,
                        previous_actuator_state=None,
                        steering_visual_gain=steering_visual_gain,
                        throttle_visual_gain=throttle_visual_gain,
                        use_kinematics_in_ue5=use_kinematics_in_ue5,
                        min_render_hz=min_render_hz,
                        max_translation_step=max_translation_step,
                        max_yaw_step_rad=max_yaw_step_rad,
                        lock_path_in_ue5=lock_path_in_ue5,
                        vehicle_object_name=vehicle_object_name,
                    )
                    if draw_ship_trail_in_ue5 and bridge.is_connected:
                        previous_trail_point = _trail_point_from_motion_state(previous_motion_state)

                    if bridge.is_connected and not vehicle_object_name:
                        parked_controls = bridge.airsim.CarControls()
                        parked_controls.throttle = 0.0
                        parked_controls.steering = 0.0
                        parked_controls.brake = 1.0
                        bridge.set_controls(parked_controls)
                    if adaptive_settle_s > 0.0:
                        time.sleep(adaptive_settle_s)
                    try:
                        bridge.get_state()
                        last_bridge_telemetry_at = time.perf_counter()
                    except Exception:
                        pass

                    total_reward = 0.0
                    step = 0
                    steps_since_ue5_update = 0
                    last_log_t = 0.0
                    episode_wall_start = time.perf_counter()
                    expected_obs_dim = getattr(policy, "input_dim", None)
                    expected_action_dim = None
                    action_space = getattr(env, "action_space", None)
                    if action_space is not None and getattr(action_space, "shape", None) is not None:
                        expected_action_dim = int(np.prod(action_space.shape, dtype=np.int64))

                    while step < max_steps:
                        policy_obs = _normalise_policy_observation(obs, expected_dim=expected_obs_dim)
                        action, _, _ = policy.predict(policy_obs, deterministic=True)
                        action = _normalise_policy_action(action, expected_dim=expected_action_dim)

                        obs, reward, done, truncated, _ = env.step(action)
                        total_reward += float(reward)
                        step += 1

                        now = time.perf_counter()
                        target_motion_state = self._motion_state_from_env(env, bridge)
                        steps_since_ue5_update += 1
                        batched_steps = steps_since_ue5_update
                        should_push_ue5 = (
                            steps_since_ue5_update >= steps_per_ue5_update
                            or done
                            or truncated
                        )
                        if should_push_ue5:
                            steps_since_ue5_update = 0
                            previous_motion_state, previous_actuator_state = self._animate_state(
                                env,
                                bridge,
                                target_motion_state=target_motion_state,
                                previous_motion_state=previous_motion_state,
                                total_duration=target_step_interval_s * max(1, batched_steps),
                                interpolation_steps=max(1, int(render_interpolation_steps)),
                                mirror_controls_in_ue5=mirror_controls_in_ue5,
                                animate_steering_in_ue5=animate_steering_in_ue5,
                                animate_throttle_in_ue5=animate_throttle_in_ue5,
                                previous_actuator_state=previous_actuator_state,
                                steering_visual_gain=steering_visual_gain,
                                throttle_visual_gain=throttle_visual_gain,
                                use_kinematics_in_ue5=use_kinematics_in_ue5,
                                min_render_hz=min_render_hz,
                                max_translation_step=max_translation_step,
                                max_yaw_step_rad=max_yaw_step_rad,
                                lock_path_in_ue5=lock_path_in_ue5,
                                vehicle_object_name=vehicle_object_name,
                            )
                            if (now - last_bridge_telemetry_at) >= bridge_telemetry_interval_s or done or truncated:
                                try:
                                    bridge.get_state()
                                    last_bridge_telemetry_at = now
                                except Exception:
                                    pass
                            if draw_ship_trail_in_ue5 and bridge.is_connected:
                                current_trail_point = _trail_point_from_motion_state(previous_motion_state)
                                if previous_trail_point is not None:
                                    moved_dx = float(current_trail_point.x_val - previous_trail_point.x_val)
                                    moved_dy = float(current_trail_point.y_val - previous_trail_point.y_val)
                                    if np.hypot(moved_dx, moved_dy) >= float(ship_trail_min_segment_m):
                                        # Only accumulate here (live drawing drops FPS);
                                        # the trail is drawn once at episode end.
                                        trail_points_history.append(current_trail_point)
                                        previous_trail_point = current_trail_point
                                else:
                                    previous_trail_point = current_trail_point
                                    trail_points_history.append(current_trail_point)
                        else:
                            previous_motion_state = target_motion_state

                        session_exporter.log_visualisation_step(
                            build_framework_visualisation_row(
                                env=env,
                                bridge=bridge,
                                episode_index=episode + 1,
                                step_index=step,
                                reward=float(reward),
                                action=action,
                                outcome="terminated" if done else ("truncated" if truncated else "running"),
                                model_origin=self.adapter.model_origin,
                            )
                        )

                        if (
                            verbose
                            and
                            log_every > 0
                            and step % log_every == 0
                            and ((now - last_log_t) >= max(float(min_log_interval_s), 0.0) or done or truncated)
                        ):
                            last_log_t = now
                            ship_x, ship_y, _ = self._extract_pose(env)
                            surge_speed, _, _ = self._extract_velocity(env)
                            controls = getattr(bridge, "current_controls", getattr(bridge, "_controls", None))
                            logger.info(
                                "step=%d pos=(%.0f, %.0f) speed=%.2f steer=%+.3f throttle=%.3f reward=%+.2f",
                                step,
                                ship_x,
                                ship_y,
                                surge_speed,
                                getattr(controls, "steering", 0.0) if controls is not None else 0.0,
                                getattr(controls, "throttle", 0.0) if controls is not None else 0.0,
                                reward,
                            )

                        if done or truncated:
                            break

                        target_elapsed_s = step * target_step_interval_s
                        actual_elapsed_s = time.perf_counter() - episode_wall_start
                        remaining_budget_s = target_elapsed_s - actual_elapsed_s
                        if remaining_budget_s > 0.001:
                            time.sleep(min(remaining_budget_s, target_step_interval_s))

                    # Episode end: draw the full trail persistently, then flush the
                    # persistent 3D lines (the minimap keeps its markers).
                    if bridge.is_connected:
                        if draw_ship_trail_in_ue5 and len(trail_points_history) >= 2:
                            trail_chunk_size = 500
                            for chunk_start in range(
                                0,
                                len(trail_points_history),
                                trail_chunk_size - 1,
                            ):
                                chunk = trail_points_history[
                                    chunk_start : chunk_start + trail_chunk_size
                                ]
                                if len(chunk) < 2:
                                    continue
                                try:
                                    bridge.plot_line_strip(
                                        chunk,
                                        color_rgba=[0.12, 0.52, 1.0, 1.0],
                                        thickness=float(ship_trail_thickness),
                                        is_persistent=True,
                                    )
                                except Exception as exc:
                                    logger.debug("Trail draw chunk failed: %s", exc)

                        # Flush persistent 3D lines; the minimap keeps earlier markers.
                        try:
                            bridge.flush_persistent_markers()
                        except Exception as exc:
                            logger.debug("Episode-end flush failed: %s", exc)

                    end_x, end_y, _ = self._extract_pose(env)
                    logger.info(
                        "   Finished: %d steps | reward=%+.1f | end=(%.1f, %.1f)",
                        step, total_reward, end_x, end_y
                    )
                    session_exporter.update_metadata(
                        {
                            "last_episode_index": episode + 1,
                            "last_episode_steps": step,
                            "last_reward": float(total_reward),
                            "last_ship_position": {
                                "x": float(end_x),
                                "y": float(end_y),
                            },
                        }
                    )

            finally:
                try:
                    if bridge is not None and bridge.is_connected:
                        bridge.set_simulation_speed(1.0)
                except Exception:
                    pass
                if env is not None:
                    self.adapter.cleanup_environment(env)
                if bridge is not None:
                    bridge.disconnect()
                logger.info("Exports saved to '%s'.", session_exporter.export_dir)
                logger.info("Visualisation complete - AirSim connection closed.")

            # Reuse persisted telemetry for BI metrics and comparison plots.
            session_exporter.save()
            session_csv_path = Path(
                getattr(
                    session_exporter,
                    "telemetry_path",
                    session_exporter.export_dir / "visualisation_session.csv",
                )
            )
            bridge_csv_path = Path(
                getattr(
                    session_exporter,
                    "bridge_log_path",
                    session_exporter.export_dir / "airsim_telemetry.csv",
                )
            )
            summary = build_run_summary_from_csv(
                session_csv_path=session_csv_path,
                bridge_csv_path=bridge_csv_path,
            )
            session_exporter.update_metadata(
                {
                    "performance_summary": summary.to_metadata_dict(),
                }
            )
            session_exporter.save()

            print()
            print(format_performance_summary("AIRSIM VISUALISATION SUMMARY", summary))

            comparison_plot = render_side_by_side_trajectory_plot(
                session_csv_path=session_csv_path,
                bridge_csv_path=bridge_csv_path,
                output_path=comparison_plot_path,
                title=f"2D vs 3D Trajectory - {resolved_model_path.name}",
            )
            if comparison_plot is not None:
                logger.info("2D vs 3D trajectory plot saved to '%s'.", comparison_plot)
                session_exporter.update_metadata(
                    {
                        "comparison_plot": comparison_plot.name,
                    }
                )
                session_exporter.save()
            else:
                logger.warning("2D vs 3D trajectory plot skipped (missing telemetry or matplotlib).")


def run_visualise_models(model_path: str, model_family: str = "auto", **kwargs) -> None:
    """
    Dispatch visualization to the selected model-family adapter.

    Args:
        model_path: Model checkpoint path used for replay.
        model_family: Requested adapter family, or "auto".
        **kwargs: Options forwarded to VisualiseModelsRunner.run.
    """

    env_name = str(kwargs.pop("env_name", "auto"))
    adapter = resolve_visualisation_adapter(
        model_family=model_family,
        model_path=model_path,
        env_name=env_name,
    )

    print(type(adapter))
    print(adapter.family)
    print(adapter.supported_environment_names)
    resolved_env_name = "FhSimEnv" if adapter.family == "fhsim" else env_name
    runner = VisualiseModelsRunner(adapter=adapter, env_name=resolved_env_name)
    runner.run(model_path=model_path, **kwargs)


def _build_parser() -> argparse.ArgumentParser:
    """
    Command-line parser for framework-model replay in UE5.
    """

    parser = argparse.ArgumentParser(
        description="Visualise framework model playback in UE5.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", required=True, help="Path to model .zip file.")
    parser.add_argument("--model-family", default="auto", choices=["auto", *get_supported_model_families()])
    parser.add_argument("--env", default="auto", choices=["auto", *get_supported_environment_names()])
    parser.add_argument("--ip", default="")
    parser.add_argument("--port", type=int, default=41451)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=14000, dest="max_steps")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--save-slot", default=DEFAULT_AIRSIM_SAVE_SLOT)
    parser.add_argument(
        "--simulation-speed",
        type=float,
        default=get_runtime_settings().get("airsim", {}).get(
            "simulation_speed", DEFAULT_AIRSIM_SIMULATION_SPEED
        ),
    )
    parser.add_argument("--mirror-controls", action="store_true", default=False)
    parser.add_argument("--no-steering-visual", action="store_true", default=False)
    parser.add_argument("--no-throttle-visual", action="store_true", default=False)
    parser.add_argument("--render-interpolation-steps", type=int, default=8)
    parser.add_argument("--steering-visual-gain", type=float, default=0.35)
    parser.add_argument("--throttle-visual-gain", type=float, default=1.0)
    parser.add_argument("--use-kinematics-in-ue5", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--min-render-hz", type=float, default=90.0)
    parser.add_argument("--max-translation-step", type=float, default=0.15)
    parser.add_argument("--max-yaw-step-deg", type=float, default=1.5)
    parser.add_argument("--lock-path-in-ue5", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--no-calibration-reference", action="store_true", default=False)
    parser.add_argument("--vehicle-object-name", default="", dest="vehicle_object_name")
    parser.add_argument("--draw-ship-trail-in-ue5", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--ship-trail-thickness", type=float, default=14.0)
    parser.add_argument("--ship-trail-min-segment-m", type=float, default=_DEFAULT_TRAIL_MIN_SEGMENT_M)
    parser.add_argument("--min-log-interval-s", type=float, default=_DEFAULT_MIN_LOG_INTERVAL_S)
    parser.add_argument("--bridge-telemetry-hz", type=float, default=_DEFAULT_BRIDGE_TELEMETRY_HZ)
    parser.add_argument("--startup-settle-s", type=float, default=_DEFAULT_STARTUP_SETTLE_S)
    parser.add_argument("--verbose", action="store_true", default=False)
    return parser


if __name__ == "__main__":
    from fw.common.tools.logging_config import setup_logging

    args = _build_parser().parse_args()

    log_path = setup_logging(
        run_type="visualize",
        env_name="airsim",
        agent_type=Path(args.model).stem,
        src_dir=_SRC_DIR,
        level=logging.INFO if args.verbose else logging.WARNING,
    )
    logger.info("Logging to file: %s", log_path)

    run_visualise_models(
        model_path=args.model,
        model_family=args.model_family,
        env_name=args.env,
        ip=args.ip,
        port=args.port,
        max_steps=args.max_steps,
        num_episodes=args.episodes,
        device=args.device,
        draw_calibration_reference=not args.no_calibration_reference,
        log_every=args.log_every,
        save_slot=args.save_slot,
        simulation_speed=args.simulation_speed,
        mirror_controls_in_ue5=args.mirror_controls,
        animate_steering_in_ue5=not args.no_steering_visual,
        animate_throttle_in_ue5=not args.no_throttle_visual,
        render_interpolation_steps=args.render_interpolation_steps,
        steering_visual_gain=args.steering_visual_gain,
        throttle_visual_gain=args.throttle_visual_gain,
        use_kinematics_in_ue5=args.use_kinematics_in_ue5,
        min_render_hz=args.min_render_hz,
        max_translation_step=args.max_translation_step,
        max_yaw_step_deg=args.max_yaw_step_deg,
        lock_path_in_ue5=args.lock_path_in_ue5,
        vehicle_object_name=args.vehicle_object_name,
        draw_ship_trail_in_ue5=args.draw_ship_trail_in_ue5,
        ship_trail_thickness=args.ship_trail_thickness,
        ship_trail_min_segment_m=args.ship_trail_min_segment_m,
        min_log_interval_s=args.min_log_interval_s,
        bridge_telemetry_hz=args.bridge_telemetry_hz,
        startup_settle_s=args.startup_settle_s,
        verbose=args.verbose,
    )
