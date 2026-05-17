"""
Stability tests for AirSim training components (BI-3.3.2).

Covers four areas:
- StrictSuccessStoppingCallback: streak logic, CTE violations, multi-env
- Reward function: bounded output across varied ship states
- Observation space: stays within declared bounds across many steps
- Training log analysis: robust against edge-case CSV data
"""

from __future__ import annotations

import csv
import textwrap
from types import SimpleNamespace

import numpy as np

from airsim.train.strict_success_callback import StrictSuccessStoppingCallback


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_callback(n_envs: int = 1, **kwargs) -> StrictSuccessStoppingCallback:
    """
    Create a callback and initialize per-env trackers directly.

    SB3's BaseCallback.training_env is a read-only property in newer releases,
    so we bypass _on_training_start and call _reset_episode_trackers directly.
    """

    cb = StrictSuccessStoppingCallback(**kwargs)
    for env_index in range(n_envs):
        cb._reset_episode_trackers(env_index)
    return cb


def _step(
    cb: StrictSuccessStoppingCallback,
    reward: float,
    done: bool,
    info: dict,
) -> bool:
    """
    Drive one fake SB3 step through the callback.
    """

    cb.locals = {
        "rewards": [float(reward)],
        "dones": [bool(done)],
        "infos": [info],
    }
    return cb._on_step()


def _success_info(*, cte: float = 0.0) -> dict:
    """
    Info dict that satisfies all strict-success criteria.
    """

    return {
        "all_checkpoints_passed": True,
        "checkpoint_index": 3,
        "route_total_checkpoints": 3,
        "cross_track_error": cte,
        "strict_cte_violation": False,
        "episode": {"r": 0.97, "l": 1},
    }


def _failure_info() -> dict:
    """
    Info dict that fails the strict-success check.
    """

    return {
        "all_checkpoints_passed": False,
        "checkpoint_index": 0,
        "route_total_checkpoints": 3,
        "cross_track_error": 5.0,
        "strict_cte_violation": False,
    }


# ---------------------------------------------------------------------------
# StrictSuccessStoppingCallback — streak logic
# ---------------------------------------------------------------------------


class TestStrictSuccessStreakLogic:
    """
    Tests for consecutive-success counter increment and reset behavior.
    """

    def test_initial_consecutive_count_is_zero(self):
        """
        Counter starts at zero before any steps are taken.
        """

        cb = _make_callback()
        assert cb._consecutive_successes == 0

    def test_strict_success_increments_counter(self):
        """
        A single successful done-step increments the counter to 1.
        """

        cb = _make_callback(min_average_episode_reward=0.95, strict_cte_limit_m=10.0)
        _step(cb, reward=0.97, done=True, info=_success_info())
        assert cb._consecutive_successes == 1

    def test_two_successes_increment_counter_twice(self):
        """
        Two consecutive successes increment the counter to 2.
        """

        cb = _make_callback(min_average_episode_reward=0.95)
        _step(cb, reward=0.97, done=True, info=_success_info())
        _step(cb, reward=0.97, done=True, info=_success_info())
        assert cb._consecutive_successes == 2

    def test_failed_episode_resets_counter_to_zero(self):
        """
        A failure after a success resets the counter back to zero.
        """

        cb = _make_callback(min_average_episode_reward=0.95)
        _step(cb, reward=0.97, done=True, info=_success_info())
        assert cb._consecutive_successes == 1

        _step(cb, reward=0.2, done=True, info=_failure_info())
        assert cb._consecutive_successes == 0

    def test_counter_resets_after_intermittent_failure(self):
        """
        Counter resets to zero on failure and can accumulate again from scratch.
        """

        cb = _make_callback(min_average_episode_reward=0.95, required_consecutive_episodes=3)
        for _ in range(2):
            _step(cb, reward=0.97, done=True, info=_success_info())
        assert cb._consecutive_successes == 2

        _step(cb, reward=0.1, done=True, info=_failure_info())
        assert cb._consecutive_successes == 0

        _step(cb, reward=0.97, done=True, info=_success_info())
        assert cb._consecutive_successes == 1

    def test_non_done_steps_do_not_change_counter(self):
        """
        Steps where done=False must not modify the consecutive-success counter.
        """

        cb = _make_callback()
        _step(cb, reward=0.5, done=False, info=_success_info())
        assert cb._consecutive_successes == 0

    def test_non_done_steps_after_success_keep_counter(self):
        """
        Non-done steps after a success leave the counter unchanged.
        """

        cb = _make_callback(min_average_episode_reward=0.95)
        _step(cb, reward=0.97, done=True, info=_success_info())
        count_after_done = cb._consecutive_successes

        _step(cb, reward=0.5, done=False, info=_success_info())
        assert cb._consecutive_successes == count_after_done


# ---------------------------------------------------------------------------
# StrictSuccessStoppingCallback — training stop
# ---------------------------------------------------------------------------


class TestStrictSuccessStopSignal:
    """
    Tests for the training-stop return value relative to the required threshold.
    """

    def test_continues_training_before_threshold(self):
        """
        _on_step returns True (continue) while below the required consecutive count.
        """

        cb = _make_callback(required_consecutive_episodes=3, min_average_episode_reward=0.95)
        for _ in range(2):
            result = _step(cb, reward=0.97, done=True, info=_success_info())
        assert result is True

    def test_stops_training_exactly_at_threshold(self):
        """
        _on_step returns False (stop) as soon as the required consecutive count is reached.
        """

        cb = _make_callback(required_consecutive_episodes=3, min_average_episode_reward=0.95)
        for _ in range(2):
            _step(cb, reward=0.97, done=True, info=_success_info())
        result = _step(cb, reward=0.97, done=True, info=_success_info())
        assert result is False

    def test_stops_training_with_threshold_of_one(self):
        """
        A threshold of 1 stops training on the very first success.
        """

        cb = _make_callback(required_consecutive_episodes=1, min_average_episode_reward=0.95)
        result = _step(cb, reward=0.97, done=True, info=_success_info())
        assert result is False

    def test_continues_training_indefinitely_without_success(self):
        """
        Training never stops when all episodes fail, regardless of count.
        """

        cb = _make_callback(required_consecutive_episodes=2, min_average_episode_reward=0.95)
        for _ in range(20):
            result = _step(cb, reward=0.2, done=True, info=_failure_info())
        assert result is True
        assert cb._consecutive_successes == 0


# ---------------------------------------------------------------------------
# StrictSuccessStoppingCallback — failure modes
# ---------------------------------------------------------------------------


class TestStrictSuccessFailureModes:
    """
    Tests for individual criteria that must block a strict-success classification.
    """

    def test_cte_above_limit_blocks_success(self):
        """
        CTE exceeding strict_cte_limit_m prevents a success from being counted.
        """

        cb = _make_callback(strict_cte_limit_m=10.0, min_average_episode_reward=0.95)
        _step(cb, reward=0.97, done=True, info=_success_info(cte=11.0))
        assert cb._consecutive_successes == 0

    def test_cte_exactly_at_limit_is_allowed(self):
        """
        CTE exactly equal to the limit is within tolerance and counts as success.
        """

        cb = _make_callback(strict_cte_limit_m=10.0, min_average_episode_reward=0.95)
        _step(cb, reward=0.97, done=True, info=_success_info(cte=10.0))
        assert cb._consecutive_successes == 1

    def test_strict_cte_violation_flag_blocks_success(self):
        """
        strict_cte_violation=True in info prevents the episode from counting as success.
        """

        cb = _make_callback(min_average_episode_reward=0.95)
        info = _success_info()
        info["strict_cte_violation"] = True
        _step(cb, reward=0.97, done=True, info=info)
        assert cb._consecutive_successes == 0

    def test_checkpoints_not_passed_blocks_success(self):
        """
        all_checkpoints_passed=False in info prevents a success from being counted.
        """

        cb = _make_callback(min_average_episode_reward=0.95)
        info = _success_info()
        info["all_checkpoints_passed"] = False
        info["checkpoint_index"] = 1
        _step(cb, reward=0.97, done=True, info=info)
        assert cb._consecutive_successes == 0

    def test_reward_below_threshold_blocks_success(self):
        """
        Episode reward below min_average_episode_reward prevents a success count.
        """

        cb = _make_callback(min_average_episode_reward=0.95)
        info = _success_info()
        info["episode"] = {"r": 0.80, "l": 1}
        _step(cb, reward=0.80, done=True, info=info)
        assert cb._consecutive_successes == 0

    def test_missing_info_keys_are_tolerated(self):
        """
        An info dict without required keys must not raise and must not stop training.
        """

        cb = _make_callback(required_consecutive_episodes=2, min_average_episode_reward=0.95)
        result = _step(cb, reward=0.0, done=True, info={})
        assert result is True

    def test_empty_infos_and_rewards_are_tolerated(self):
        """
        Empty rewards/dones/infos lists must not raise and must return True.
        """

        cb = _make_callback()
        cb.locals = {"rewards": [], "dones": [], "infos": []}
        assert cb._on_step() is True


# ---------------------------------------------------------------------------
# StrictSuccessStoppingCallback — multi-env
# ---------------------------------------------------------------------------


class TestStrictSuccessMultiEnv:
    """
    Tests for callback behavior when multiple parallel environments are active.
    """

    def _make_multi_callback(self, n_envs: int, **kwargs) -> StrictSuccessStoppingCallback:
        """
        Create a callback pre-initialised for n_envs parallel environments.
        """

        return _make_callback(n_envs=n_envs, **kwargs)

    def _multi_step(
        self,
        cb: StrictSuccessStoppingCallback,
        rewards: list[float],
        dones: list[bool],
        infos: list[dict],
    ) -> bool:
        """
        Drive one fake SB3 step with per-env reward/done/info vectors.
        """

        cb.locals = {
            "rewards": rewards,
            "dones": dones,
            "infos": infos,
        }
        return cb._on_step()

    def test_failure_after_success_in_same_step_resets_counter(self):
        """
        Envs are processed sequentially: a failure that follows a success in the same
        step still resets the counter because the callback iterates env_0 then env_1.
        """

        cb = self._make_multi_callback(2, min_average_episode_reward=0.95)
        self._multi_step(
            cb,
            rewards=[0.97, 0.1],
            dones=[True, True],
            infos=[_success_info(), _failure_info()],
        )
        assert cb._consecutive_successes == 0

    def test_failure_in_any_env_resets_streak(self):
        """
        A failure in the last env of a step resets the counter to zero even if
        earlier envs in the same step succeeded.
        """

        cb = self._make_multi_callback(2, min_average_episode_reward=0.95)
        self._multi_step(
            cb,
            rewards=[0.97, 0.97],
            dones=[True, True],
            infos=[_success_info(), _success_info()],
        )
        assert cb._consecutive_successes == 2

        self._multi_step(
            cb,
            rewards=[0.97, 0.1],
            dones=[True, True],
            infos=[_success_info(), _failure_info()],
        )
        assert cb._consecutive_successes == 0

    def test_all_envs_succeed_increments_by_n_envs(self):
        """
        Counter increases by n_envs when every parallel env succeeds in the same step.
        """

        cb = self._make_multi_callback(3, min_average_episode_reward=0.95)
        self._multi_step(
            cb,
            rewards=[0.97, 0.97, 0.97],
            dones=[True, True, True],
            infos=[_success_info(), _success_info(), _success_info()],
        )
        assert cb._consecutive_successes == 3


# ---------------------------------------------------------------------------
# Reward function stability
# ---------------------------------------------------------------------------


class _FakeCarControls:
    """
    Minimal stand-in for airsim.CarControls used in fake bridge tests.
    """

    def __init__(self) -> None:
        """
        Initialize all control fields to their neutral/default values.
        """

        self.steering = 0.0
        self.throttle = 0.0
        self.brake = 0.0
        self.is_manual_gear = False
        self.manual_gear = 0
        self.gear_immediate = False


class _FakeQuaternion:
    """
    Minimal quaternion encoding a pure yaw rotation for pose stubs.
    """

    def __init__(self, yaw: float) -> None:
        """
        Compute quaternion components from a yaw angle in radians.
        """

        self.w_val = float(np.cos(yaw / 2.0))
        self.x_val = 0.0
        self.y_val = 0.0
        self.z_val = float(np.sin(yaw / 2.0))


class _FakeBridge:
    """
    In-process stub replacing AirSimBridge for environment unit tests.
    """

    def __init__(self, config) -> None:
        """
        Initialize the fake bridge with a config object and zero-state fields.
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
        Record the save slot and increment the load counter.
        """

        self.save_slot = save_slot
        self.load_world_save_calls += 1

    def set_simulation_speed(self, simulation_speed: float) -> None:
        """
        Store the requested simulation speed multiplier.
        """

        self.simulation_speed = float(simulation_speed)

    def set_rendering_enabled(self, is_enabled: bool, force: bool = False) -> bool:
        """
        Store the rendering flag and return True to indicate success.
        """

        self.rendering_enabled = bool(is_enabled)
        return True

    def set_pause(self, is_paused: bool) -> None:
        """
        Set the internal paused state.
        """

        self.paused = bool(is_paused)

    def is_paused(self) -> bool:
        """
        Return the current paused state.
        """

        return self.paused

    def continue_for_time(self, seconds: float) -> None:
        """
        Advance the fake simulation by a duration in seconds.
        """

        self.last_continue_mode = "time"
        self.last_continue_value = float(seconds)
        self._advance()

    def continue_for_frames(self, frames: int) -> None:
        """
        Advance the fake simulation by a fixed number of frames.
        """

        self.last_continue_mode = "frames"
        self.last_continue_value = int(frames)
        self._advance()

    def _advance(self) -> None:
        """
        Update kinematic state for one tick based on current controls.
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
        Store the control input for use in the next _advance call.
        """

        self.current_controls = controls

    def get_state(self) -> dict:
        """
        Return the current kinematic state or a failure sentinel dict.
        """

        if self.fail_state_reads:
            return {"state_read_failed": True, "error": "synthetic read failure"}
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
        Return collision state or raise a RuntimeError when fail_collision_reads is set.
        """

        if self.fail_collision_reads:
            raise RuntimeError("synthetic collision read failure")
        return SimpleNamespace(has_collided=self.collision)

    def get_vehicle_pose(self):
        """
        Return a pose namespace or raise a RuntimeError when fail_pose_reads is set.
        """

        if self.fail_pose_reads:
            raise RuntimeError("synthetic pose failure")
        return SimpleNamespace(
            position=SimpleNamespace(x_val=self.pose_x, y_val=self.pose_y, z_val=self.render_z),
            orientation=_FakeQuaternion(self.yaw),
        )

    def set_pose(self, x, y, z, yaw, preserve_current_z=False, preserve_roll_pitch=False) -> None:
        """
        Update the vehicle's position and heading in the fake world.
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
        Reset dynamic state and increment the reset counter.
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


def _patch_env(monkeypatch):
    """
    Monkeypatch AirSimEnv to use _FakeBridge and pre-built route/outline arrays.
    """

    from airsim.communication.coord_transform import CoordTransform
    import airsim.environments.airsim_env as airsim_env_module

    identity_transform = CoordTransform(
        {"scale": 1.0, "rotation": 0.0, "framework_origin": [0.0, 0.0], "airsim_origin": [0.0, 0.0]}
    )
    route = np.array([[0.0, 0.0], [250.0, 0.0], [500.0, 0.0]], dtype=np.float32)
    outline = np.array(
        [[-50.0, -50.0], [250.0, -50.0], [250.0, 50.0], [-50.0, 50.0]], dtype=np.float32
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


class TestRewardStability:
    """
    Tests that the reward function stays finite and correctly ordered across varied inputs.
    """

    def test_normalized_reward_bounded_after_many_steps(self, monkeypatch):
        """
        Normalized reward must stay in [-1, 1] across 50 consecutive random steps.
        """

        _patch_env(monkeypatch)
        from airsim.environments.airsim_env import AirSimEnv

        rng = np.random.default_rng(42)
        env = AirSimEnv(
            ship_pos=[0.0, 0.0],
            target_pos=[500.0, 0.0],
            normalize_reward=True,
            use_pause_stepping=True,
            use_sim_time_stepping=True,
        )
        env.reset()

        for _ in range(50):
            action = rng.uniform(-1.0, 1.0, size=(2,)).astype(np.float32)
            _, reward, terminated, truncated, _ = env.step(action)
            assert -1.0 <= reward <= 1.0, f"Reward {reward} out of bounds"
            if terminated or truncated:
                env.reset()

        env.close()

    def test_reward_finite_for_extreme_cte(self, monkeypatch):
        """
        Reward must be a finite number even for very large cross-track errors.
        """

        _patch_env(monkeypatch)
        from airsim.environments.airsim_env import AirSimEnv

        env = AirSimEnv(
            ship_pos=[0.0, 0.0],
            target_pos=[500.0, 0.0],
            normalize_reward=True,
            use_pause_stepping=True,
            use_sim_time_stepping=True,
        )
        env.reset()
        env.step_count = 10

        for extreme_cte in [500.0, 1000.0, 5000.0]:
            prev = {"cross_track_error": extreme_cte - 1.0, "route_progress_m": 10.0}
            curr = {"cross_track_error": extreme_cte, "route_progress_m": 11.0}
            result = env.calculate_reward(previous_state=prev, current_state=curr, checkpoint_reward=0.0)
            raw_reward = result[0] if isinstance(result, tuple) else result
            assert np.isfinite(raw_reward), f"Non-finite reward for CTE={extreme_cte}"

        env.close()

    def test_reward_better_centered_than_offtrack(self, monkeypatch):
        """
        Centered tracking must consistently score higher than wide off-track tracking.
        """

        _patch_env(monkeypatch)
        from airsim.environments.airsim_env import AirSimEnv

        env = AirSimEnv(
            ship_pos=[0.0, 0.0],
            target_pos=[500.0, 0.0],
            normalize_reward=False,
            use_pause_stepping=True,
            use_sim_time_stepping=True,
        )
        env.reset()
        env.step_count = 10
        env.ship_speed = 5.6
        env.ship_yaw_rate = 0.0
        env.heading_error = np.deg2rad(2.0)
        env.performed_action = np.array([0.0, 1.0], dtype=np.float32)
        env.previous_action = np.array([0.0, 1.0], dtype=np.float32)
        env.last_policy_action = np.array([0.0, 1.0], dtype=np.float32)
        env._route_progress_m = 120.0

        centered = env.calculate_reward(
            previous_state={"cross_track_error": 1.0, "route_progress_m": 119.0},
            current_state={"cross_track_error": 2.0, "route_progress_m": 120.0},
            checkpoint_reward=0.0,
        )
        wide = env.calculate_reward(
            previous_state={"cross_track_error": 120.0, "route_progress_m": 119.0},
            current_state={"cross_track_error": 150.0, "route_progress_m": 120.0},
            checkpoint_reward=0.0,
        )

        centered_val = centered[0] if isinstance(centered, tuple) else centered
        wide_val = wide[0] if isinstance(wide, tuple) else wide
        assert centered_val > wide_val

        env.close()


# ---------------------------------------------------------------------------
# Observation space stability
# ---------------------------------------------------------------------------


class TestObservationSpaceStability:
    """
    Tests that observations stay within declared space bounds after resets and steps.
    """

    def test_observations_always_in_bounds_across_many_steps(self, monkeypatch):
        """
        All observations over 40 steps with random actions must satisfy observation_space.
        """

        _patch_env(monkeypatch)
        from airsim.environments.airsim_env import AirSimEnv

        rng = np.random.default_rng(7)
        env = AirSimEnv(
            ship_pos=[0.0, 0.0],
            target_pos=[500.0, 0.0],
            use_pause_stepping=True,
            use_sim_time_stepping=True,
        )
        obs, _ = env.reset()
        assert env.observation_space.contains(obs), "Initial obs out of bounds"

        for step in range(40):
            action = rng.uniform(-1.0, 1.0, size=(2,)).astype(np.float32)
            obs, _, terminated, truncated, _ = env.step(action)
            assert env.observation_space.contains(obs), f"Obs out of bounds at step {step}: {obs}"
            if terminated or truncated:
                obs, _ = env.reset()
                assert env.observation_space.contains(obs), "Post-reset obs out of bounds"

        env.close()

    def test_observations_correct_shape_after_multiple_resets(self, monkeypatch):
        """
        Observation shape must be (18,) after five resets.
        """

        _patch_env(monkeypatch)
        from airsim.environments.airsim_env import AirSimEnv

        env = AirSimEnv(
            ship_pos=[0.0, 0.0],
            target_pos=[500.0, 0.0],
            use_pause_stepping=True,
            use_sim_time_stepping=True,
        )
        for _ in range(5):
            obs, _ = env.reset()
            assert obs.shape == (18,), f"Unexpected shape: {obs.shape}"
            assert obs.dtype == np.float32

        env.close()

    def test_observation_dtype_is_float32(self, monkeypatch):
        """
        Observations must be float32 as declared by the observation space.
        """

        _patch_env(monkeypatch)
        from airsim.environments.airsim_env import AirSimEnv

        env = AirSimEnv(
            ship_pos=[0.0, 0.0],
            target_pos=[500.0, 0.0],
            use_pause_stepping=True,
            use_sim_time_stepping=True,
        )
        obs, _ = env.reset()
        assert obs.dtype == np.float32

        obs, _, _, _, _ = env.step(np.array([0.2, 0.8], dtype=np.float32))
        assert obs.dtype == np.float32

        env.close()


# ---------------------------------------------------------------------------
# Training log analysis stability
# ---------------------------------------------------------------------------


class TestTrainingLogAnalysisStability:
    """
    Tests that training log analysis handles edge-case CSV inputs without raising.
    """

    def test_returns_safe_defaults_for_nonexistent_file(self, tmp_path):
        """
        Missing log file should return zero-episode defaults without raising.
        """

        from airsim.train.training_log_analysis import analyze_training_log

        result = analyze_training_log(tmp_path / "missing.csv")

        assert result.total_rows == 0
        assert result.episode_count == 0
        assert result.best_max_checkpoint == 0
        assert result.high_speed_offtrack_ratio == 0.0

    def test_returns_safe_defaults_for_empty_csv(self, tmp_path):
        """
        An empty CSV (header only) must not raise and must return zero counts.
        """

        from airsim.train.training_log_analysis import analyze_training_log

        log = tmp_path / "empty.csv"
        log.write_text("episode,checkpoint_index,speed,cross_track_error\n", encoding="utf-8")

        result = analyze_training_log(log)

        assert result.total_rows == 0
        assert result.episode_count == 0

    def test_skips_malformed_rows_without_raising(self, tmp_path):
        """
        Rows with missing or non-numeric columns must be silently skipped.
        """

        from airsim.train.training_log_analysis import analyze_training_log

        log = tmp_path / "malformed.csv"
        log.write_text(
            textwrap.dedent("""\
                episode,checkpoint_index,speed,cross_track_error
                bad,row,here,oops
                1,2,4.0,5.0
                ,,,
                2,3,5.0,10.0
            """),
            encoding="utf-8",
        )

        result = analyze_training_log(log)

        assert result.episode_count == 2
        assert result.total_rows == 2

    def test_curriculum_recommendation_within_route_bounds(self, tmp_path):
        """
        Curriculum checkpoint recommendation must never exceed actual route length.
        """

        from airsim.train.training_log_analysis import analyze_training_log

        log = tmp_path / "route.csv"
        rows = [("episode", "checkpoint_index", "speed", "cross_track_error")]
        for ep in range(20):
            for step in range(10):
                rows.append((ep, min(step + 1, 5), 4.0, 3.0))

        with log.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerows(rows)

        route_total = 5
        result = analyze_training_log(log, route_total_checkpoints=route_total)

        assert result.recommended_curriculum_target_checkpoint <= route_total
        assert result.recommended_curriculum_window_size <= route_total

    def test_high_speed_offtrack_ratio_between_zero_and_one(self, tmp_path):
        """
        High-speed off-track ratio must always be a valid fraction.
        """

        from airsim.train.training_log_analysis import analyze_training_log

        log = tmp_path / "ratio.csv"
        rows = [("episode", "checkpoint_index", "speed", "cross_track_error")]
        for ep in range(5):
            rows.append((ep, 1, 5.5, 80.0))
            rows.append((ep, 2, 2.0, 5.0))

        with log.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerows(rows)

        result = analyze_training_log(log)

        assert 0.0 <= result.high_speed_offtrack_ratio <= 1.0

    def test_best_checkpoint_matches_maximum_in_log(self, tmp_path):
        """
        best_max_checkpoint should equal the highest checkpoint index seen.
        """

        from airsim.train.training_log_analysis import analyze_training_log

        log = tmp_path / "best.csv"
        rows = [("episode", "checkpoint_index", "speed", "cross_track_error")]
        for ep, max_cp in enumerate([2, 5, 3, 7, 1]):
            for cp in range(1, max_cp + 1):
                rows.append((ep, cp, 3.0, 5.0))

        with log.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerows(rows)

        result = analyze_training_log(log)

        assert result.best_max_checkpoint == 7
