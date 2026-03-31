"""Modular model visualisation runner for UE5."""

from __future__ import annotations

import argparse
import logging
import math
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

_SRC_DIR = Path(__file__).resolve().parent.parent.parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from airsim.communication.airsim_bridge import AirSimBridge  # noqa: E402
from airsim.config.airsim_config import (  # noqa: E402
    AirSimConfig,
    DEFAULT_AIRSIM_SAVE_SLOT,
    DEFAULT_AIRSIM_SIMULATION_SPEED,
)
from airsim.inference.minimap_models_ue5 import (  # noqa: E402
    UE5TopDownMinimap,
    VisualiseModelsMinimap,
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
from fw.common.airsim.tools.path_utils import resolve_model_path  # noqa: E402

logger = logging.getLogger(__name__)

_DEFAULT_MIN_LOG_INTERVAL_S = 0.25
_DEFAULT_MINIMAP_UPDATE_HZ = 20.0
_DEFAULT_BRIDGE_TELEMETRY_HZ = 12.0
_DEFAULT_STARTUP_SETTLE_S = 0.4
_DEFAULT_TRAIL_MIN_SEGMENT_M = 1.5


class VisualiseModelsRunner:
    """Framework-physics visualiser that mirrors model output into UE5."""

    def __init__(self, adapter: ModelVisualisationAdapter, env_name: str = "auto") -> None:
        """Initialize the generic runner with a model-family adapter."""
        self.adapter = adapter
        self.env_name = env_name

    def _resolve_environment_name(self, model_path: str) -> str:
        """Resolve which framework environment should back the visualisation."""
        return self.adapter.resolve_environment_name(model_path, self.env_name)

    def _load_policy(self, model_path: str, device: str):
        """Load the framework policy used for replay."""
        return self.adapter.load_policy(model_path, device=device)

    def _build_environment(self, selected_env: str, max_steps: int):
        """Construct the framework environment for replay."""
        return self.adapter.build_environment(selected_env, max_steps=max_steps)

    def _motion_state_from_env(self, env, bridge: AirSimBridge):
        """Extract the current framework motion state."""
        return self.adapter.motion_state_from_env(env, bridge)

    def _animate_state(self, env, bridge: AirSimBridge, **kwargs):
        """Mirror one framework state transition into UE5."""
        return self.adapter.animate_state(env, bridge, **kwargs)

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
        throttle_visual_gain: float = 1.0,
        use_kinematics_in_ue5: bool = False,
        min_render_hz: float = 90.0,
        max_translation_step: float = 0.15,
        max_yaw_step_deg: float = 1.5,
        lock_path_in_ue5: bool = False,
        vehicle_object_name: str = "",
        show_minimap: bool = True,
        use_ue5_camera_minimap: bool = False,
        ue5_camera_name: str = "TopDownCam",
        ue5_camera_fov_deg: float = 38.0,
        ue5_camera_height_m: float = 120.0,
        ue5_camera_follow_heading: bool = False,
        ue5_preview_window: bool = True,
        save_matplotlib_minimap: bool = False,
        draw_ship_trail_in_ue5: bool = True,
        ship_trail_thickness: float = 14.0,
        ship_trail_min_segment_m: float = _DEFAULT_TRAIL_MIN_SEGMENT_M,
        min_log_interval_s: float = _DEFAULT_MIN_LOG_INTERVAL_S,
        minimap_update_hz: float = _DEFAULT_MINIMAP_UPDATE_HZ,
        bridge_telemetry_hz: float = _DEFAULT_BRIDGE_TELEMETRY_HZ,
        startup_settle_s: float = _DEFAULT_STARTUP_SETTLE_S,
    ) -> None:
        """Run framework-model playback mirrored into UE5.

        Args:
            model_path: Path to the trained model checkpoint.
            ip: AirSim server IP address.
            port: AirSim RPC port.
            max_steps: Maximum number of framework steps per episode.
            num_episodes: Number of episodes to visualise.
            device: Inference device passed to the policy loader.
            draw_calibration_reference: Whether to draw the calibration overlay in UE5.
            log_every: Log every N steps when logging is enabled.
            save_slot: AirSim save slot to load before playback.
            simulation_speed: UE5 simulation-speed multiplier.
            mirror_controls_in_ue5: Whether to mirror framework controls into UE5.
            animate_steering_in_ue5: Whether to animate steering while mirroring.
            animate_throttle_in_ue5: Whether to animate throttle while mirroring.
            render_interpolation_steps: Number of interpolation steps per UE5 update.
            steering_visual_gain: Steering visualisation gain.
            throttle_visual_gain: Throttle visualisation gain.
            use_kinematics_in_ue5: Whether to move the UE5 actor kinematically.
            min_render_hz: Minimum UE5 render update rate target.
            max_translation_step: Maximum translation distance per interpolation step.
            max_yaw_step_deg: Maximum yaw delta per interpolation step in degrees.
            lock_path_in_ue5: Whether to lock motion onto the framework path.
            vehicle_object_name: Optional UE5 vehicle object override.
            show_minimap: Whether to create a minimap during playback.
            use_ue5_camera_minimap: Whether to use the UE5 top-down camera minimap.
            ue5_camera_name: UE5 camera name for the top-down minimap.
            ue5_camera_fov_deg: UE5 top-down camera field of view.
            ue5_camera_height_m: UE5 top-down camera follow height.
            ue5_camera_follow_heading: Whether the UE5 minimap camera follows heading.
            ue5_preview_window: Whether to show the UE5 camera preview window.
            save_matplotlib_minimap: Whether to save the matplotlib minimap snapshot when supported.
            draw_ship_trail_in_ue5: Whether to draw a persistent blue trail behind the ship in UE5.
            ship_trail_thickness: Thickness of the persistent blue ship trail in UE5.
            ship_trail_min_segment_m: Minimum movement before adding a new UE5 trail segment.
            min_log_interval_s: Minimum wall-clock interval between info logs.
            minimap_update_hz: Maximum minimap refresh rate.
            bridge_telemetry_hz: Maximum bridge telemetry polling rate.
            startup_settle_s: Startup settle window before stepping begins.
        """
        selected_env = self._resolve_environment_name(model_path)
        resolved_model_path = resolve_model_path(model_path, _SRC_DIR)

        banner = "=" * 62
        print(banner)
        print("  MODEL VISUALISATION IN UE5")
        print(f"  Model      : {resolved_model_path.name}")
        print(f"  Family     : {self.adapter.family}")
        print(f"  Environment: {selected_env}")
        print(f"  Save slot  : {save_slot}")
        print(f"  Camera map : {'UE5 top-down' if use_ue5_camera_minimap else 'matplotlib minimap'}")
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
                    "show_minimap": show_minimap,
                    "use_ue5_camera_minimap": use_ue5_camera_minimap,
                    "ue5_camera_name": ue5_camera_name,
                    "ue5_camera_fov_deg": ue5_camera_fov_deg,
                    "ue5_camera_height_m": ue5_camera_height_m,
                    "ue5_camera_follow_heading": ue5_camera_follow_heading,
                    "ue5_preview_window": ue5_preview_window,
                    "save_matplotlib_minimap": save_matplotlib_minimap,
                    "draw_ship_trail_in_ue5": draw_ship_trail_in_ue5,
                    "ship_trail_thickness": float(ship_trail_thickness),
                    "ship_trail_min_segment_m": float(ship_trail_min_segment_m),
                    "minimap_update_hz": float(minimap_update_hz),
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
            bridge = None
            env = None
            minimap = None

            def _trail_point_from_motion_state(motion_state: np.ndarray):
                """Return the current UE5 trail point for the ship path marker."""
                trail_offset_m = 8.0
                trail_x = float(motion_state[0] - np.cos(float(motion_state[2])) * trail_offset_m)
                trail_y = float(motion_state[1] - np.sin(float(motion_state[2])) * trail_offset_m)
                render_z = float(getattr(bridge, "render_z", 0.0))
                trail_z = float(render_z - 12.0)
                vector3r_factory = getattr(bridge._airsim, "Vector3r", None)
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
                minimap_update_interval_s = 1.0 / max(float(minimap_update_hz), 1.0)
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
                last_minimap_update_at = 0.0
                last_bridge_telemetry_at = 0.0

                bridge.prepare_model_visualisation(
                    save_slot=save_slot,
                    simulation_speed=simulation_speed,
                    draw_reference=draw_calibration_reference,
                )

                if not mirror_controls_in_ue5 and not animate_steering_in_ue5 and not animate_throttle_in_ue5:
                    neutral_controls = bridge._airsim.CarControls()
                    neutral_controls.throttle = 0.0
                    neutral_controls.steering = 0.0
                    neutral_controls.brake = 0.0
                    bridge.set_controls(neutral_controls)

                for episode in range(num_episodes):
                    print(f"\n-- Visualisation episode {episode + 1}/{num_episodes} --")
                    obs, _ = env.reset()
                    previous_trail_point = None

                    if show_minimap:
                        if minimap is not None:
                            minimap.close()
                        try:
                            if use_ue5_camera_minimap:
                                minimap = UE5TopDownMinimap(
                                    bridge,
                                    camera_name=ue5_camera_name,
                                    fov_degrees=ue5_camera_fov_deg,
                                    height_offset_m=ue5_camera_height_m,
                                    follow_heading=ue5_camera_follow_heading,
                                    enable_preview_window=ue5_preview_window,
                                )
                            else:
                                minimap = VisualiseModelsMinimap(
                                    env,
                                    save_dir=str(session_exporter.export_dir),
                                    max_fps=float(minimap_update_hz),
                                )
                        except Exception as exc:
                            logger.warning("Minimap unavailable: %s", exc)
                            minimap = None

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
                    if draw_ship_trail_in_ue5:
                        previous_trail_point = _trail_point_from_motion_state(previous_motion_state)

                    if bridge.is_connected and not vehicle_object_name:
                        parked_controls = bridge._airsim.CarControls()
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

                    while step < max_steps:
                        policy_obs = obs[0] if isinstance(obs, np.ndarray) and obs.ndim > 1 else obs
                        action, _, _ = policy.predict(policy_obs, deterministic=True)
                        action = np.asarray(action).reshape(-1)

                        obs, reward, done, truncated, _ = env.step(action)
                        total_reward += float(reward)
                        step += 1

                        now = time.perf_counter()
                        if minimap is not None and (now - last_minimap_update_at) >= minimap_update_interval_s:
                            last_minimap_update_at = now
                            minimap.update(env)

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
                            if draw_ship_trail_in_ue5:
                                current_trail_point = _trail_point_from_motion_state(previous_motion_state)
                                if previous_trail_point is not None:
                                    moved_dx = float(current_trail_point.x_val - previous_trail_point.x_val)
                                    moved_dy = float(current_trail_point.y_val - previous_trail_point.y_val)
                                    if np.hypot(moved_dx, moved_dy) >= float(ship_trail_min_segment_m):
                                        bridge.plot_line_strip(
                                            [previous_trail_point, current_trail_point],
                                            color_rgba=[0.12, 0.52, 1.0, 1.0],
                                            thickness=float(ship_trail_thickness),
                                            is_persistent=True,
                                        )
                                        previous_trail_point = current_trail_point
                                else:
                                    previous_trail_point = current_trail_point
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
                            log_every > 0
                            and step % log_every == 0
                            and ((now - last_log_t) >= max(float(min_log_interval_s), 0.0) or done or truncated)
                        ):
                            last_log_t = now
                            logger.info(
                                "step=%d pos=(%.0f, %.0f) speed=%.2f steer=%+.3f throttle=%.3f reward=%+.2f",
                                step,
                                env.ship_pos[0][0],
                                env.ship_pos[0][1],
                                env.phys_sims[0].state.u,
                                bridge._controls.steering if bridge._controls is not None else 0.0,
                                bridge._controls.throttle if bridge._controls is not None else 0.0,
                                reward,
                            )

                        if done or truncated:
                            break

                        target_elapsed_s = step * target_step_interval_s
                        actual_elapsed_s = time.perf_counter() - episode_wall_start
                        remaining_budget_s = target_elapsed_s - actual_elapsed_s
                        if remaining_budget_s > 0.001:
                            time.sleep(min(remaining_budget_s, target_step_interval_s))

                    print(
                        f"   Finished: {step} steps | reward={total_reward:+.1f} | "
                        f"end=({env.ship_pos[0][0]:.1f}, {env.ship_pos[0][1]:.1f})"
                    )
                    session_exporter.update_metadata(
                        {
                            "last_episode_index": episode + 1,
                            "last_episode_steps": step,
                            "last_reward": float(total_reward),
                            "last_ship_position": {
                                "x": float(env.ship_pos[0][0]),
                                "y": float(env.ship_pos[0][1]),
                            },
                        }
                    )

                    if save_matplotlib_minimap and minimap is not None and hasattr(minimap, "finish"):
                        minimap.finish(total_reward, episode=episode + 1)
            finally:
                if minimap is not None:
                    minimap.close()
                try:
                    if bridge is not None and bridge.is_connected:
                        bridge.set_simulation_speed(1.0)
                except Exception:
                    pass
                if env is not None:
                    self.adapter.cleanup_environment(env)
                if bridge is not None:
                    bridge.disconnect()
                print(f"Exports saved to '{session_exporter.export_dir}'.")
                print("\nVisualisation complete - AirSim connection closed.")
def run_visualise_models(model_path: str, model_family: str = "auto", **kwargs) -> None:
    """Dispatch visualization to the selected model-family adapter."""
    env_name = str(kwargs.pop("env_name", "auto"))
    adapter = resolve_visualisation_adapter(
        model_family=model_family,
        model_path=model_path,
        env_name=env_name,
    )
    resolved_env_name = "FhSimEnv" if adapter.family == "fhsim" else env_name
    runner = VisualiseModelsRunner(adapter=adapter, env_name=resolved_env_name)
    runner.run(model_path=model_path, **kwargs)


def _build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for framework-model replay."""
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
    parser.add_argument("--simulation-speed", type=float, default=15.0)
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
    parser.add_argument("--no-minimap", action="store_true", default=False, dest="no_minimap")
    parser.add_argument("--use-ue5-camera-minimap", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--ue5-camera-name", default="TopDownCam")
    parser.add_argument("--ue5-camera-fov-deg", type=float, default=38.0)
    parser.add_argument("--ue5-camera-height-m", type=float, default=120.0)
    parser.add_argument("--ue5-camera-follow-heading", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--ue5-preview-window", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-matplotlib-minimap", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--draw-ship-trail-in-ue5", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--ship-trail-thickness", type=float, default=14.0)
    parser.add_argument("--ship-trail-min-segment-m", type=float, default=_DEFAULT_TRAIL_MIN_SEGMENT_M)
    parser.add_argument("--min-log-interval-s", type=float, default=_DEFAULT_MIN_LOG_INTERVAL_S)
    parser.add_argument("--minimap-update-hz", type=float, default=_DEFAULT_MINIMAP_UPDATE_HZ)
    parser.add_argument("--bridge-telemetry-hz", type=float, default=_DEFAULT_BRIDGE_TELEMETRY_HZ)
    parser.add_argument("--startup-settle-s", type=float, default=_DEFAULT_STARTUP_SETTLE_S)
    return parser


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    args = _build_parser().parse_args()
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
        show_minimap=not args.no_minimap,
        use_ue5_camera_minimap=args.use_ue5_camera_minimap,
        ue5_camera_name=args.ue5_camera_name,
        ue5_camera_fov_deg=args.ue5_camera_fov_deg,
        ue5_camera_height_m=args.ue5_camera_height_m,
        ue5_camera_follow_heading=args.ue5_camera_follow_heading,
        ue5_preview_window=args.ue5_preview_window,
        save_matplotlib_minimap=args.save_matplotlib_minimap,
        draw_ship_trail_in_ue5=args.draw_ship_trail_in_ue5,
        ship_trail_thickness=args.ship_trail_thickness,
        ship_trail_min_segment_m=args.ship_trail_min_segment_m,
        min_log_interval_s=args.min_log_interval_s,
        minimap_update_hz=args.minimap_update_hz,
        bridge_telemetry_hz=args.bridge_telemetry_hz,
        startup_settle_s=args.startup_settle_s,
    )
