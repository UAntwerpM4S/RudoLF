"""Modular model visualisation runner for UE5.

This entrypoint generalizes the previous Fossen-only visualiser so multiple
framework model families can reuse the same UE5 mirroring pipeline.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

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
from airsim.inference.visualisation_models.run_visualise_fossen import (  # noqa: E402
    _animate_framework_state,
    _build_fossen_environment,
    _detect_fossen_environment,
    _framework_motion_state_from_env,
    load_policy,
)
from airsim.inference.minimap_models_ue5 import (  # noqa: E402
    UE5TopDownMinimap,
    VisualiseModelsMinimap,
)

logger = logging.getLogger(__name__)


class VisualiseFossen:
    """Framework-physics visualiser that mirrors model output into UE5."""

    def __init__(self, env_name: str = "auto") -> None:
        """Initialize the instance.

        Args:
            env_name: Description.
        """
        self.env_name = env_name

    def _resolve_environment_name(self, model_path: str) -> str:
        """ resolve environment name.

        Args:
            model_path: Description.
        """
        return _detect_fossen_environment(model_path, self.env_name)

    def _build_environment(self, selected_env: str, max_steps: int):
        """ build environment.

        Args:
            selected_env: Description.
            max_steps: Description.
        """
        return _build_fossen_environment(selected_env, max_steps)

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
    ) -> None:
        """Run.

        Args:
            model_path: Description.
            ip: Description.
            port: Description.
            max_steps: Description.
            num_episodes: Description.
            device: Description.
            draw_calibration_reference: Description.
            log_every: Description.
            save_slot: Description.
            simulation_speed: Description.
            mirror_controls_in_ue5: Description.
            animate_steering_in_ue5: Description.
            animate_throttle_in_ue5: Description.
            render_interpolation_steps: Description.
            steering_visual_gain: Description.
            throttle_visual_gain: Description.
            use_kinematics_in_ue5: Description.
            min_render_hz: Description.
            max_translation_step: Description.
            max_yaw_step_deg: Description.
            lock_path_in_ue5: Description.
            vehicle_object_name: Description.
            show_minimap: Description.
            use_ue5_camera_minimap: Description.
            ue5_camera_name: Description.
            ue5_camera_fov_deg: Description.
            ue5_camera_height_m: Description.
            ue5_camera_follow_heading: Description.
            ue5_preview_window: Description.
            ,: Description.
        """
        selected_env = self._resolve_environment_name(model_path)
        banner = "=" * 62
        print(banner)
        print("  MODEL VISUALISATION IN UE5")
        print(f"  Model      : {Path(model_path).name}")
        print(f"  Framework  : {selected_env}")
        print(f"  Save slot  : {save_slot}")
        print(f"  Camera map : {'UE5 top-down' if use_ue5_camera_minimap else 'matplotlib minimap'}")
        print(banner)

        policy = load_policy(model_path, device=device)
        bridge = AirSimBridge(AirSimConfig(ip=ip, port=port))
        if not bridge.is_connected:
            raise ConnectionError(
                f"AirSimBridge could not connect to UE5 at {ip or '127.0.0.1'}:{port}."
            )

        env = self._build_environment(selected_env, max_steps=max_steps)
        max_yaw_step_rad = np.deg2rad(max_yaw_step_deg)
        minimap = None

        try:
            bridge.prepare_fossen_visualisation(
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
                            minimap = VisualiseModelsMinimap(env, save_dir=".")
                    except Exception as exc:
                        logger.warning("Minimap unavailable: %s", exc)
                        minimap = None

                previous_motion_state = _framework_motion_state_from_env(env, bridge)
                previous_actuator_state = None
                previous_motion_state, previous_actuator_state = _animate_framework_state(
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

                if bridge.is_connected and not vehicle_object_name:
                    _park = bridge._airsim.CarControls()
                    _park.throttle = 0.0
                    _park.steering = 0.0
                    _park.brake = 1.0
                    bridge.set_controls(_park)
                time.sleep(0.4)

                total_reward = 0.0
                step = 0
                _ue5_update_dt = 1.0 / min_render_hz if min_render_hz > 0.0 else 0.0
                _last_ue5_t = time.perf_counter() - _ue5_update_dt

                while step < max_steps:
                    policy_obs = obs[0] if isinstance(obs, np.ndarray) and obs.ndim > 1 else obs
                    action, _, _ = policy.predict(policy_obs, deterministic=True)
                    action = np.asarray(action).reshape(-1)

                    obs, reward, done, truncated, _ = env.step(action)
                    total_reward += float(reward)
                    step += 1

                    if minimap is not None:
                        minimap.update(env)

                    target_motion_state = _framework_motion_state_from_env(env, bridge)
                    now = time.perf_counter()
                    if (now - _last_ue5_t) >= _ue5_update_dt or done or truncated:
                        _last_ue5_t = now
                        previous_motion_state, previous_actuator_state = _animate_framework_state(
                            env,
                            bridge,
                            target_motion_state=target_motion_state,
                            previous_motion_state=previous_motion_state,
                            total_duration=env.time_step / max(simulation_speed, 1e-6),
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
                    else:
                        previous_motion_state = target_motion_state

                    if log_every > 0 and step % log_every == 0:
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

                print(
                    f"   Finished: {step} steps | reward={total_reward:+.1f} | "
                    f"end=({env.ship_pos[0][0]:.1f}, {env.ship_pos[0][1]:.1f})"
                )

                if minimap is not None and hasattr(minimap, "finish"):
                    minimap.finish(total_reward, episode=episode + 1)
        finally:
            if minimap is not None:
                minimap.close()
            try:
                if bridge.is_connected:
                    bridge.set_simulation_speed(1.0)
            except Exception:
                pass
            if hasattr(env, "fh_sim"):
                env.fh_sim.dispose()
            bridge.disconnect()
            print("\nVisualisation complete - AirSim connection closed.")


class VisualiseFhSim(VisualiseFossen):
    """FhSim-specialized visualiser with a fixed framework environment."""

    def __init__(self) -> None:
        """Initialize the instance.
        """
        super().__init__(env_name="FhSimEnv")


def run_visualise_models(model_path: str, model_family: str = "auto", **kwargs) -> None:
    """Dispatch visualization to the selected model visualiser class."""
    family = str(model_family).strip().lower()
    if family == "fhsim":
        runner = VisualiseFhSim()
    else:
        env_name = str(kwargs.pop("env_name", "auto"))
        runner = VisualiseFossen(env_name=env_name)
    runner.run(model_path=model_path, **kwargs)


def _build_parser() -> argparse.ArgumentParser:
    """ build parser.
    """
    parser = argparse.ArgumentParser(
        description="Visualise framework model playback in UE5.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", required=True, help="Path to model .zip file.")
    parser.add_argument("--model-family", default="auto", choices=["auto", "fossen", "fhsim"])
    parser.add_argument("--env", default="auto", choices=["auto", "PySimEnv", "FhSimEnv"])
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
    )

