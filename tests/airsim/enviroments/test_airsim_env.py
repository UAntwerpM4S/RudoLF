"""
Tests for AirSimEnv (with a fake AirSim bridge).
"""

from __future__ import annotations

import pytest
import numpy as np

from types import SimpleNamespace
from airsim.communication.coord_transform import CoordTransform


class _FakeCarControls:
    """
    Tiny stand-in for cosysairsim.CarControls.
    """

    def __init__(self) -> None:
        """
        Initialize neutral control values.
        """

        self.steering = 0.0
        self.throttle = 0.0
        self.brake = 0.0
        self.is_manual_gear = False
        self.manual_gear = 0
        self.gear_immediate = False


class _FakeQuaternion:
    """
    Small quaternion helper storing a planar yaw rotation.
    """

    def __init__(self, yaw: float) -> None:
        """
        Build a zero-roll, zero-pitch quaternion from yaw.
        """

        self.w_val = float(np.cos(yaw / 2.0))
        self.x_val = 0.0
        self.y_val = 0.0
        self.z_val = float(np.sin(yaw / 2.0))


class _FakeBridge:
    """
    Small AirSim bridge double for deterministic env tests.
    """

    def __init__(self, config) -> None:
        """
        Store a couple of runtime fields used by AirSimEnv.
        """

        self.config = config
        self.is_connected = True
        self.airsim = SimpleNamespace(CarControls=_FakeCarControls)
        self.client = object()
        self.coord_transform = None
        self.render_z = 0.0
        self.simulation_speed = 1.0
        self.paused = False
        self.current_controls = None
        self.collision = False
        self.x = 0.0
        self.y = 0.0
        self.pose_x = 0.0
        self.pose_y = 0.0
        self.yaw = 0.0
        self.ang_vel_z = 0.0
        self.speed = 0.0
        self.timestamp = 0
        self.last_continue_mode = None
        self.last_continue_value = None
        self.pose_updates_state = True
        self.pause_after_advance = True
        self.freeze_progress = False
        self.fail_state_reads = False
        self.fail_pose_reads = False
        self.fail_collision_reads = False
        self.reset_calls = 0
        self.load_world_save_calls = 0

    def load_world_save(self, save_slot: str) -> None:
        """
        Record the active save slot.
        """

        self.save_slot = save_slot
        self.load_world_save_calls += 1

    def set_simulation_speed(self, simulation_speed: float) -> None:
        """
        Record the requested simulation speed.
        """

        self.simulation_speed = float(simulation_speed)

    def set_rendering_enabled(self, is_enabled: bool, force: bool = False) -> bool:
        """
        Store the requested render state.
        """

        self.rendering_enabled = bool(is_enabled)
        return True

    def set_pause(self, is_paused: bool) -> None:
        """
        Toggle the fake paused state.
        """

        self.paused = bool(is_paused)

    def is_paused(self) -> bool:
        """
        Return the fake paused state.
        """

        return self.paused

    def continue_for_time(self, seconds: float) -> None:
        """
        Advance the fake vehicle for one sim-time step.
        """

        self.last_continue_mode = "time"
        self.last_continue_value = float(seconds)
        self._advance()

    def continue_for_frames(self, frames: int) -> None:
        """
        Advance the fake vehicle for one frame step.
        """

        self.last_continue_mode = "frames"
        self.last_continue_value = int(frames)
        self._advance()

    def _advance(self) -> None:
        """
        Move the fake vehicle forward based on throttle and steering.
        """

        if self.freeze_progress:
            self.paused = bool(self.pause_after_advance)
            return
        throttle = float(getattr(self.current_controls, "throttle", 0.0))
        steering = float(getattr(self.current_controls, "steering", 0.0))
        self.ang_vel_z = steering * 0.15
        self.yaw += self.ang_vel_z
        self.speed = throttle * 6.0
        self.pose_x += throttle * 120.0
        self.pose_y += 0.0
        self.x = self.pose_x
        self.y = self.pose_y
        self.timestamp += 1
        self.paused = bool(self.pause_after_advance)

    def set_controls(self, controls) -> None:
        """
        Store the latest control object.
        """

        self.current_controls = controls

    def get_state(self) -> dict:
        """
        Return a telemetry payload matching AirSimBridge.get_state().
        """

        if self.fail_state_reads:
            return {
                "state_read_failed": True,
                "error": "synthetic read failure",
            }
        return {
            "pos_x": self.x,
            "pos_y": self.y,
            "speed": self.speed,
            "yaw": self.yaw,
            "ang_vel_z": self.ang_vel_z,
            "timestamp": self.timestamp,
        }

    def get_collision_info(self):
        """
        Return a minimal collision-info payload.
        """

        if self.fail_collision_reads:
            raise RuntimeError("synthetic collision read failure")
        return SimpleNamespace(has_collided=self.collision)

    def get_vehicle_pose(self):
        """
        Return a minimal pose object with position and orientation.
        """

        if self.fail_pose_reads:
            raise RuntimeError("synthetic pose failure")
        return SimpleNamespace(
            position=SimpleNamespace(
                x_val=self.pose_x,
                y_val=self.pose_y,
                z_val=self.render_z,
            ),
            orientation=_FakeQuaternion(self.yaw),
        )

    def set_pose(
        self,
        x: float,
        y: float,
        z: float,
        yaw: float,
        preserve_current_z: bool = False,
        preserve_roll_pitch: bool = False,
    ) -> None:
        """
        Teleport the fake vehicle.
        """

        self.pose_x = float(x)
        self.pose_y = float(y)
        if self.pose_updates_state:
            self.x = self.pose_x
            self.y = self.pose_y
        self.render_z = float(z)
        self.yaw = float(yaw)

    def reset(self) -> None:
        """
        Reset the fake vehicle timing state.
        """

        self.reset_calls += 1
        self.speed = 0.0
        self.ang_vel_z = 0.0
        self.paused = True

    def disconnect(self) -> None:
        """
        Mark the bridge as disconnected.
        """

        self.is_connected = False


def _patch_env_dependencies(monkeypatch) -> None:
    """
    Install the fake bridge and simple route assets for AirSimEnv tests.
    """

    import airsim.environments.airsim_env as airsim_env_module

    identity_transform = CoordTransform(
        {
            "scale": 1.0,
            "rotation": 0.0,
            "framework_origin": [0.0, 0.0],
            "airsim_origin": [0.0, 0.0],
        }
    )
    route = np.array([[0.0, 0.0], [250.0, 0.0], [500.0, 0.0]], dtype=np.float32)
    outline = np.array(
        [[-50.0, -50.0], [250.0, -50.0], [250.0, 50.0], [-50.0, 50.0]],
        dtype=np.float32,
    )

    monkeypatch.setattr(airsim_env_module, "AirSimBridge", _FakeBridge)
    monkeypatch.setattr(
        airsim_env_module,
        "load_saved_calibration_reference",
        lambda *args, **kwargs: (identity_transform, {}),
    )
    monkeypatch.setattr(
        airsim_env_module.AirSimEnv,
        "_load_route_points",
        lambda self, target_pos=None: route.copy(),
    )
    monkeypatch.setattr(
        airsim_env_module.AirSimEnv,
        "_load_outline_points",
        lambda self, path: outline.copy(),
    )


def test_reset_returns_valid_ppo_observation(monkeypatch):
    """
    AirSimEnv should expose a fixed PPO-friendly observation space.
    """

    _patch_env_dependencies(monkeypatch)
    from airsim.environments.airsim_env import AirSimEnv

    env = AirSimEnv(
        ship_pos=[0.0, 0.0],
        target_pos=[500.0, 0.0],
        use_pause_stepping=True,
        use_sim_time_stepping=True,
    )
    obs, info = env.reset()

    assert env.action_space.shape == (2,)
    assert env.observation_space.shape == (18,)
    assert obs.dtype == np.float32
    assert env.observation_space.contains(obs)
    assert info["outcome"] == "reset"
    assert info["mdp_step_seconds"] == 0.1
    assert info["guidance_assist_enabled"] is False
    assert info["control_interface"] == "car_physics"
    assert info["observation_profile"] == "ppo_route_relative_car"
    assert env.checkpoint_index == 1

    env.close()


def test_step_advances_route_with_pause_stepping(monkeypatch):
    """
    A normal step should move the ship forward and keep PPO outputs usable.
    """

    _patch_env_dependencies(monkeypatch)
    from airsim.environments.airsim_env import AirSimEnv

    env = AirSimEnv(
        ship_pos=[0.0, 0.0],
        target_pos=[500.0, 0.0],
        use_pause_stepping=True,
        use_sim_time_stepping=True,
        sim_time_per_action=0.2,
    )
    env.reset()

    obs, reward, terminated, truncated, info = env.step(
        np.array([0.0, 1.0], dtype=np.float32)
    )

    assert not terminated
    assert not truncated
    assert env.ship_pos[0] > 0.0
    assert env.last_controls["throttle"] > 0.0
    assert env.bridge.last_continue_mode == "time"
    assert env.bridge.last_continue_value == 0.2
    assert env.observation_space.contains(obs)
    assert -1.0 <= reward <= 1.0
    assert info["checkpoint_index"] >= 1
    assert "cross_track_error_delta" in info
    assert "yaw_rate" in info
    assert info["state_update_reliable"] is True
    assert info["repeated_state_count"] == 0
    assert info["reward"] == pytest.approx(reward)
    assert info["raw_reward"] != pytest.approx(reward)

    env.close()


def test_step_accepts_progress_when_pause_flag_does_not_return(monkeypatch):
    """
    Paused sim-time stepping should still succeed when telemetry advances but simIsPause stays false.
    """

    _patch_env_dependencies(monkeypatch)
    from airsim.environments.airsim_env import AirSimEnv

    env = AirSimEnv(
        ship_pos=[0.0, 0.0],
        target_pos=[500.0, 0.0],
        use_pause_stepping=True,
        use_sim_time_stepping=True,
        physics_update_timeout=0.05,
        state_poll_interval=0.001,
    )
    env.reset()
    env.bridge.pause_after_advance = False

    obs, reward, terminated, truncated, info = env.step(
        np.array([0.0, 1.0], dtype=np.float32)
    )

    assert env.observation_space.contains(obs)
    assert reward != 0.0
    assert terminated is False
    assert truncated is False
    assert info["pause_timeout"] is False

    env.close()


def test_steering_dynamics_rate_limit_blocks_snap_reversal(monkeypatch):
    """
    Steering dynamics should bound abrupt left-right rudder reversals.
    """

    _patch_env_dependencies(monkeypatch)
    from airsim.environments.airsim_env import AirSimEnv

    env = AirSimEnv(
        ship_pos=[0.0, 0.0],
        target_pos=[500.0, 0.0],
        use_pause_stepping=True,
        use_sim_time_stepping=True,
        sim_time_per_action=0.1,
        steering_smoothing_alpha=1.0,
        max_steering_rate_per_second=2.0,
    )
    env.reset()

    first = env._compose_effective_action(np.array([1.0, 1.0], dtype=np.float32))
    env.performed_action = first.copy()
    second = env._compose_effective_action(np.array([-1.0, 1.0], dtype=np.float32))

    assert first[0] == pytest.approx(0.2)
    assert second[0] == pytest.approx(0.0, abs=1e-6)
    assert first[1] == pytest.approx(1.0)
    assert second[1] == pytest.approx(1.0)
    assert env.last_steering_rate_limited is True

    env.close()


def test_step_reaches_target_and_terminates(monkeypatch):
    """
    AirSimEnv should terminate successfully when the route target is reached.
    """

    _patch_env_dependencies(monkeypatch)
    from airsim.environments.airsim_env import AirSimEnv

    env = AirSimEnv(
        ship_pos=[0.0, 0.0],
        target_pos=[500.0, 0.0],
        use_pause_stepping=True,
        use_sim_time_stepping=True,
    )
    env.reset()

    for _ in range(4):
        _, reward, terminated, truncated, info = env.step(
            np.array([0.0, 1.0], dtype=np.float32)
        )
        if terminated:
            break

    assert terminated is True
    assert truncated is False
    assert reward > 0.0
    assert info["outcome"] == "success"

    env.close()


def test_default_rewind_reset_does_not_call_bridge_reset(monkeypatch):
    """
    The default reset mode should rewind by pose without the heavier bridge reset.
    """

    _patch_env_dependencies(monkeypatch)
    from airsim.environments.airsim_env import AirSimEnv

    env = AirSimEnv(
        ship_pos=[0.0, 0.0],
        target_pos=[500.0, 0.0],
        reset_mode="rewind_pose",
        use_sim_time_stepping=True,
    )

    env.reset()
    env.reset()

    assert env.bridge.reset_calls == 0

    env.close()


def test_step_times_out_when_pause_mode_reports_no_progress(monkeypatch):
    """
    Paused stepping should truncate when physics never advances.
    """

    _patch_env_dependencies(monkeypatch)
    from airsim.environments.airsim_env import AirSimEnv

    env = AirSimEnv(
        ship_pos=[0.0, 0.0],
        target_pos=[500.0, 0.0],
        use_pause_stepping=True,
        use_sim_time_stepping=True,
        physics_update_timeout=0.02,
        state_poll_interval=0.001,
    )
    env.reset()
    env.bridge.freeze_progress = True

    obs, reward, terminated, truncated, info = env.step(
        np.array([0.0, 1.0], dtype=np.float32)
    )

    assert env.observation_space.contains(obs)
    assert terminated is False
    assert truncated is True
    assert reward < 0.0
    assert info["pause_timeout"] is True

    env.close()


def test_step_truncates_when_no_valid_state_snapshot_is_available(monkeypatch):
    """
    A failed AirSim state read should truncate instead of using fake zero telemetry.
    """

    _patch_env_dependencies(monkeypatch)
    from airsim.environments.airsim_env import AirSimEnv

    env = AirSimEnv(
        ship_pos=[0.0, 0.0],
        target_pos=[500.0, 0.0],
        use_pause_stepping=True,
        use_sim_time_stepping=True,
    )
    env.reset()
    env.bridge.fail_state_reads = True
    env.bridge.fail_pose_reads = True

    obs, reward, terminated, truncated, info = env.step(
        np.array([0.0, 1.0], dtype=np.float32)
    )

    assert env.observation_space.contains(obs)
    assert terminated is False
    assert truncated is True
    assert reward < 0.0
    assert info["outcome"] == "state_unavailable"
    assert info["state_update_reliable"] is False

    env.close()


def test_reset_raises_when_post_teleport_state_never_becomes_reliable(monkeypatch):
    """
    Reset should fail fast when AirSim cannot provide a valid initial state.
    """

    _patch_env_dependencies(monkeypatch)
    from airsim.environments.airsim_env import AirSimEnv

    env = AirSimEnv(
        ship_pos=[0.0, 0.0],
        target_pos=[500.0, 0.0],
        use_pause_stepping=True,
        use_sim_time_stepping=True,
    )
    env.bridge.fail_state_reads = True
    env.bridge.fail_pose_reads = True

    with pytest.raises(RuntimeError, match="initial simulator state|post-teleport state"):
        env.reset()

    env.close()


def test_step_truncates_when_collision_info_cannot_be_read(monkeypatch):
    """
    Collision read failures should truncate instead of pretending there was no collision.
    """

    _patch_env_dependencies(monkeypatch)
    from airsim.environments.airsim_env import AirSimEnv

    env = AirSimEnv(
        ship_pos=[0.0, 0.0],
        target_pos=[500.0, 0.0],
        training_mode=False,
    )
    env.reset()
    env.bridge.fail_collision_reads = True

    obs, reward, terminated, truncated, info = env.step(
        np.array([0.0, 1.0], dtype=np.float32)
    )

    assert env.observation_space.contains(obs)
    assert terminated is False
    assert truncated is True
    assert reward < 0.0
    assert info["outcome"] == "collision_unavailable"

    env.close()


def test_training_mode_does_not_reload_world_on_start(monkeypatch):
    """
    Training should default to in-session startup without simRequestLoad().
    """

    _patch_env_dependencies(monkeypatch)
    from airsim.environments.airsim_env import AirSimEnv

    env = AirSimEnv(
        ship_pos=[0.0, 0.0],
        target_pos=[500.0, 0.0],
        training_mode=True,
        save_slot="slot_two",
        use_sim_time_stepping=True,
    )

    env.reset()

    assert env.bridge.load_world_save_calls == 0

    env.close()


def test_visual_mode_still_loads_world_on_start(monkeypatch):
    """
    Non-training AirSimEnv instances should still load the configured save slot.
    """

    _patch_env_dependencies(monkeypatch)
    from airsim.environments.airsim_env import AirSimEnv

    env = AirSimEnv(
        ship_pos=[0.0, 0.0],
        target_pos=[500.0, 0.0],
        training_mode=False,
        save_slot="slot_two",
        use_pause_stepping=False,
    )

    env.reset()

    assert env.bridge.load_world_save_calls == 1

    env.close()


def test_training_mode_captures_live_startup_pose_without_world_reload(monkeypatch):
    """
    Training should adopt the live simulator spawn pose as the reset anchor.
    """

    _patch_env_dependencies(monkeypatch)
    from airsim.environments.airsim_env import AirSimEnv

    env = AirSimEnv(
        ship_pos=[0.0, 0.0],
        target_pos=[500.0, 0.0],
        training_mode=True,
        load_world_on_start=False,
        capture_live_start_pose=True,
        use_sim_time_stepping=True,
    )
    env.bridge.pose_x = 15.0
    env.bridge.pose_y = -25.0
    env.bridge.render_z = -2.0
    env.bridge.yaw = 0.35

    obs, info = env.reset()

    assert env.observation_space.contains(obs)
    assert info["outcome"] == "reset"
    assert env.bridge.load_world_save_calls == 0
    assert env._captured_startup_pose_airsim is not None
    assert np.allclose(env.initial_ship_pos, np.array([15.0, -25.0], dtype=np.float32))
    assert np.allclose(env.ship_pos, np.array([15.0, -25.0], dtype=np.float32))
    assert env.ship_heading == pytest.approx(0.35)
    assert env.bridge.pose_x == pytest.approx(15.0)
    assert env.bridge.pose_y == pytest.approx(-25.0)

    env.close()


def test_training_mode_loads_world_once_then_respawns_from_captured_anchor(monkeypatch):
    """
    Training can one-shot load a save slot, capture that pose, and reuse it on later resets.
    """

    _patch_env_dependencies(monkeypatch)
    from airsim.environments.airsim_env import AirSimEnv

    env = AirSimEnv(
        ship_pos=[0.0, 0.0],
        target_pos=[500.0, 0.0],
        training_mode=True,
        save_slot="simulator_1",
        load_world_on_start=True,
        capture_live_start_pose=True,
        use_sim_time_stepping=True,
    )
    env.bridge.pose_x = 30.0
    env.bridge.pose_y = -45.0
    env.bridge.render_z = 2.0
    env.bridge.yaw = 0.4

    obs, info = env.reset()
    events = env.drain_run_events()
    event_types = [event["event_type"] for event in events]

    assert env.observation_space.contains(obs)
    assert info["outcome"] == "reset"
    assert env.bridge.load_world_save_calls == 1
    assert env._captured_startup_pose_airsim is not None
    assert np.allclose(env.initial_ship_pos, np.array([30.0, -45.0], dtype=np.float32))
    assert "world_loaded" in event_types
    assert "startup_anchor_captured" in event_types

    env.bridge.pose_x = 999.0
    env.bridge.pose_y = 999.0
    env.reset()

    assert env.bridge.load_world_save_calls == 1
    assert np.allclose(env.ship_pos, np.array([30.0, -45.0], dtype=np.float32))

    env.close()


def test_training_run_events_include_startup_and_respawn(monkeypatch):
    """
    AirSimEnv should expose queued startup and respawn diagnostics for PPO logging.
    """

    _patch_env_dependencies(monkeypatch)
    from airsim.environments.airsim_env import AirSimEnv

    env = AirSimEnv(
        ship_pos=[0.0, 0.0],
        target_pos=[500.0, 0.0],
        training_mode=True,
        load_world_on_start=False,
        capture_live_start_pose=True,
        use_sim_time_stepping=True,
    )
    env.bridge.pose_x = 12.0
    env.bridge.pose_y = -8.0
    env.bridge.render_z = -1.5
    env.bridge.yaw = 0.2

    env.reset()
    events = env.drain_run_events()
    event_types = [event["event_type"] for event in events]

    assert "startup_anchor_captured" in event_types
    assert "world_prepared" in event_types
    assert "respawn" in event_types
    assert env.drain_run_events() == []

    env.close()


def test_training_run_events_include_checkpoint_and_episode_end(monkeypatch):
    """
    Checkpoint crossings and episode outcomes should be exposed in the run-event buffer.
    """

    _patch_env_dependencies(monkeypatch)
    from airsim.environments.airsim_env import AirSimEnv

    env = AirSimEnv(
        ship_pos=[0.0, 0.0],
        target_pos=[500.0, 0.0],
        use_pause_stepping=True,
        use_sim_time_stepping=True,
    )
    env.reset()
    env.drain_run_events()

    for _ in range(4):
        _, _, terminated, _, _ = env.step(np.array([0.0, 1.0], dtype=np.float32))
        if terminated:
            break
    events = env.drain_run_events()
    event_types = [event["event_type"] for event in events]

    assert "checkpoint_passed" in event_types
    assert "episode_finished" in event_types

    env.close()


def test_bridge_reset_mode_invokes_bridge_reset(monkeypatch):
    """
    Optional bridge-reset mode should still be available for heavier rewinds.
    """

    _patch_env_dependencies(monkeypatch)
    from airsim.environments.airsim_env import AirSimEnv

    env = AirSimEnv(
        ship_pos=[0.0, 0.0],
        target_pos=[500.0, 0.0],
        reset_mode="bridge_reset",
        use_sim_time_stepping=True,
    )

    env.reset()
    env.reset()

    assert env.bridge.reset_calls == 2

    env.close()


def test_reward_normalization_can_be_disabled(monkeypatch):
    """
    Callers should be able to inspect the raw shaped reward directly.
    """

    _patch_env_dependencies(monkeypatch)
    from airsim.environments.airsim_env import AirSimEnv

    env = AirSimEnv(
        ship_pos=[0.0, 0.0],
        target_pos=[500.0, 0.0],
        normalize_reward=False,
        use_pause_stepping=True,
        use_sim_time_stepping=True,
    )
    env.reset()

    _, reward, _, _, info = env.step(np.array([0.0, 1.0], dtype=np.float32))

    assert reward == pytest.approx(info["raw_reward"])
    assert info["reward"] == pytest.approx(info["raw_reward"])

    env.close()


def test_reward_penalizes_wide_fast_checkpoint_farming(monkeypatch):
    """
    Wide, fast, worsening tracking should score worse than centered tracking.
    """

    _patch_env_dependencies(monkeypatch)
    from airsim.environments.airsim_env import AirSimEnv

    env = AirSimEnv(
        ship_pos=[0.0, 0.0],
        target_pos=[500.0, 0.0],
        normalize_reward=False,
        use_pause_stepping=True,
        use_sim_time_stepping=True,
    )
    env.reset()
    env._route_progress_m = 120.0
    env.performed_action = np.array([0.0, 1.0], dtype=np.float32)
    env.previous_action = np.array([0.0, 1.0], dtype=np.float32)
    env.last_policy_action = np.array([0.0, 1.0], dtype=np.float32)

    env.ship_speed = 5.6
    env.ship_yaw_rate = 0.0
    env.heading_error = np.deg2rad(4.0)
    env.cross_error = 18.0
    prev_centered = {"cross_track_error": 16.0, "route_progress_m": 118.0}
    curr_centered = {"cross_track_error": 18.0, "route_progress_m": 120.0}
    centered_reward = env.calculate_reward(
        previous_state=prev_centered,
        current_state=curr_centered,
        checkpoint_reward=8.0,
    )

    prev_wide = {"cross_track_error": 140.0, "route_progress_m": 118.0}
    curr_wide = {"cross_track_error": 180.0, "route_progress_m": 120.0}
    wide_reward = env.calculate_reward(
        previous_state=prev_wide,
        current_state=curr_wide,
        checkpoint_reward=8.0,
    )

    assert centered_reward > wide_reward

    env.close()


def test_offtrack_limit_anneals_towards_stricter_tracking(monkeypatch):
    """
    The off-track threshold should tighten over episodes near checkpoints.
    """

    _patch_env_dependencies(monkeypatch)
    from airsim.environments.airsim_env import AirSimEnv

    env = AirSimEnv(
        ship_pos=[0.0, 0.0],
        target_pos=[500.0, 0.0],
        use_pause_stepping=True,
        use_sim_time_stepping=True,
    )
    env.reset()
    env.checkpoint_index = len(env.checkpoints) - 2
    env.ship_pos = env.checkpoints[-1]["pos"].copy()
    env.ship_pos[0] -= 5.0

    env._episode_index = 1
    early_limit = env._effective_offtrack_limit_m()

    env._episode_index = 2000
    late_limit = env._effective_offtrack_limit_m()

    assert early_limit > late_limit
    assert early_limit == pytest.approx(150.0)
    assert late_limit == pytest.approx(90.0)

    env.close()


def test_calculate_reward_offtrack_penalty_is_shaped(monkeypatch):
    """
    Off-track reward should remain strongly negative without hard clipping to -1.
    """

    _patch_env_dependencies(monkeypatch)
    from airsim.environments.airsim_env import AirSimEnv

    env = AirSimEnv(
        ship_pos=[0.0, 0.0],
        target_pos=[500.0, 0.0],
        normalize_reward=True,
        reward_normalization_scale=5.0,
        use_pause_stepping=True,
        use_sim_time_stepping=True,
    )
    env.reset()
    env.step_count = 20

    previous_state = {"cross_track_error": 20.0, "route_progress_m": 100.0}
    current_state = {"cross_track_error": 220.0, "route_progress_m": 101.0}

    raw_reward, offtrack = env.calculate_reward(
        previous_state=previous_state,
        current_state=current_state,
        checkpoint_reward=0.0,
    )

    assert offtrack is True
    assert raw_reward <= -3.5
    assert -1.0 < env._normalize_reward(raw_reward) < -0.5

    env.close()


def test_step_terminates_on_collision(monkeypatch):
    """
    Collisions should terminate the episode with collision metadata.
    """

    _patch_env_dependencies(monkeypatch)
    from airsim.environments.airsim_env import AirSimEnv

    env = AirSimEnv(
        ship_pos=[0.0, 0.0],
        target_pos=[500.0, 0.0],
        training_mode=False,
    )
    env.reset()
    env.bridge.collision = True

    _, reward, terminated, truncated, info = env.step(
        np.array([0.0, 1.0], dtype=np.float32)
    )

    assert terminated is True
    assert truncated is False
    assert reward < 0.0
    assert info["outcome"] == "collision"

    env.close()


def test_reset_prefers_vehicle_pose_when_car_state_lags(monkeypatch):
    """
    Reset should succeed even when CarState still reports the old spawn pose.
    """

    _patch_env_dependencies(monkeypatch)
    from airsim.environments.airsim_env import AirSimEnv

    env = AirSimEnv(
        ship_pos=[0.0, 0.0],
        target_pos=[500.0, 0.0],
        use_pause_stepping=True,
        use_sim_time_stepping=True,
    )
    env.bridge.x = 4000.0
    env.bridge.y = -2500.0
    env.bridge.pose_updates_state = False

    obs, info = env.reset()

    assert env.observation_space.contains(obs)
    assert info["outcome"] == "reset"
    assert np.linalg.norm(env.ship_pos - np.array([0.0, 0.0], dtype=np.float32)) < 1e-3
    assert abs(env.cross_error) < 1e-3

    env.close()


def test_training_mode_rejects_pause_frame_stepping(monkeypatch):
    """
    Training should use an explicit sim-time RL step, not ambiguous frame stepping.
    """

    _patch_env_dependencies(monkeypatch)
    from airsim.environments.airsim_env import AirSimEnv

    with pytest.raises(ValueError, match="use_sim_time_stepping=True"):
        AirSimEnv(
            ship_pos=[0.0, 0.0],
            target_pos=[500.0, 0.0],
            training_mode=True,
            use_pause_stepping=True,
            use_sim_time_stepping=False,
        )
