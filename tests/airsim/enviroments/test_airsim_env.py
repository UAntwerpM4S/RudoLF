"""Unit tests for AirSimEnv (T-009 to T-012).

All tests are fully mocked — no UE5 or CosysAirSim is required.
"""

import csv
import importlib
import sys
from pathlib import Path
import pytest
import numpy as np
from unittest.mock import MagicMock, patch

_SRC_DIR = Path(__file__).resolve().parents[3] / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from airsim.communication.coord_transform import CoordTransform


# ── Shared helpers ─────────────────────────────────────────


def _make_mock_airsim_module():
    """Create a mock ``cosysairsim`` module with the minimum API surface."""
    m = MagicMock()

    # CarClient
    client = MagicMock()
    m.CarClient = MagicMock(return_value=client)
    m.CarControls = MagicMock

    # Pose helpers
    m.Pose = MagicMock
    m.Vector3r = MagicMock
    m.Quaternionr = MagicMock

    # simGetVehiclePose returns a pose at the origin
    pose_mock = MagicMock()
    pose_mock.position.x_val = 0.0
    pose_mock.position.y_val = 0.0
    pose_mock.position.z_val = -1.0
    pose_mock.orientation.w_val = 1.0
    pose_mock.orientation.x_val = 0.0
    pose_mock.orientation.y_val = 0.0
    pose_mock.orientation.z_val = 0.0
    client.simGetVehiclePose.return_value = pose_mock

    # getCarState returns a state with speed 0
    state_mock = MagicMock()
    state_mock.speed = 0.0
    state_mock.gear = 1
    state_mock.timestamp = 0
    kin = MagicMock()
    kin.position = pose_mock.position
    kin.linear_velocity.x_val = 0.0
    kin.linear_velocity.y_val = 0.0
    kin.linear_velocity.z_val = 0.0
    kin.angular_velocity.x_val = 0.0
    kin.angular_velocity.y_val = 0.0
    kin.angular_velocity.z_val = 0.0
    kin.orientation = pose_mock.orientation
    state_mock.kinematics_estimated = kin
    client.getCarState.return_value = state_mock

    # Collision info: no collision by default
    collision_mock = MagicMock()
    collision_mock.has_collided = False
    client.simGetCollisionInfo.return_value = collision_mock

    return m, client


def _make_env(mock_module, **env_kwargs):
    """Build an ``AirSimEnv`` with cosysairsim fully mocked.

    The real AirSimEnv constructor waits for a simulated UE5 reload cycle.
    In tests we short-circuit that sequence by controlling reachability checks
    and making sleeps return immediately.
    """
    sys.modules["cosysairsim"] = mock_module

    bridge_module = importlib.import_module("airsim.communication.airsim_bridge")
    env_module = importlib.import_module("airsim.environments.airsim_env")

    reachability_sequence = [True, False, True, True]

    with patch.object(
        bridge_module,
        "is_ue5_reachable",
        side_effect=reachability_sequence,
    ), patch.object(bridge_module.time, "sleep", return_value=None), patch.object(env_module.time, "sleep", return_value=None):
        from airsim.environments.airsim_env import AirSimEnv

        default_kwargs = {
            "ship_pos": [2245, -47],
            "target_pos": [-1559, 11990],
            "max_steps": 100,
            "use_pause_stepping": False,
            "training_mode": False,
            "verbose": False,
        }
        default_kwargs.update(env_kwargs)

        env = AirSimEnv(**default_kwargs)
    return env


def _set_reward_state(
    env,
    progress_m=10.0,
    offset_m=0.0,
    speed_mps=1.0,
    checkpoint_index=1,
    heading_offset_rad=0.0,
    throttle=0.5,
    steering=0.0,
    yaw_rate=None,
):
    """Place the mock ship on the first trajectory segment for reward tests."""
    prev_pos = env.checkpoints[max(0, checkpoint_index - 1)]["pos"]
    cur_pos = env.checkpoints[checkpoint_index]["pos"]
    direction = cur_pos - prev_pos
    unit = direction / np.linalg.norm(direction)
    perp = np.array([-unit[1], unit[0]], dtype=np.float32)
    heading = float(np.arctan2(direction[1], direction[0])) + float(heading_offset_rad)

    env.checkpoint_index = checkpoint_index
    env.previous_ship_pos = prev_pos + unit * 10.0 + perp * offset_m
    env.ship_pos = env.previous_ship_pos + unit * progress_m
    env.previous_heading = float(np.arctan2(direction[1], direction[0]))
    env.ship_heading = heading
    env.ship_speed = speed_mps
    env.ship_r = (
        float(yaw_rate)
        if yaw_rate is not None
        else float(heading_offset_rad / max(env.time_step, 1e-9))
    )
    env.previous_action = np.array([0.0, throttle], dtype=np.float32)
    env.performed_action = np.array([steering, throttle], dtype=np.float32)
    env.last_controls = {"steering": float(steering), "throttle": float(throttle), "brake": 0.0}
    env._ship_in_open_water = lambda pos: True
    env._check_collision = lambda: False


def _find_turn_checkpoint(env, min_turn_deg=20.0):
    """Return a checkpoint index where the route turns by at least ``min_turn_deg``."""
    for checkpoint_index in range(1, len(env.checkpoints) - 1):
        prev_pos = env.checkpoints[checkpoint_index - 1]["pos"]
        cur_pos = env.checkpoints[checkpoint_index]["pos"]
        next_pos = env.checkpoints[checkpoint_index + 1]["pos"]
        current_heading = float(np.arctan2(*(cur_pos - prev_pos)[::-1]))
        next_heading = float(np.arctan2(*(next_pos - cur_pos)[::-1]))
        turn_angle = abs(float(np.arctan2(np.sin(next_heading - current_heading), np.cos(next_heading - current_heading))))
        if turn_angle >= np.deg2rad(min_turn_deg):
            return checkpoint_index
    return None


def _find_straight_checkpoint(env, max_turn_deg=5.0):
    """Return a checkpoint index where the route turn angle stays below ``max_turn_deg``."""
    for checkpoint_index in range(1, len(env.checkpoints) - 1):
        prev_pos = env.checkpoints[checkpoint_index - 1]["pos"]
        cur_pos = env.checkpoints[checkpoint_index]["pos"]
        next_pos = env.checkpoints[checkpoint_index + 1]["pos"]
        current_heading = float(np.arctan2(*(cur_pos - prev_pos)[::-1]))
        next_heading = float(np.arctan2(*(next_pos - cur_pos)[::-1]))
        turn_angle = abs(
            float(
                np.arctan2(
                    np.sin(next_heading - current_heading),
                    np.cos(next_heading - current_heading),
                )
            )
        )
        if turn_angle <= np.deg2rad(max_turn_deg):
            return checkpoint_index
    return None


# ── Fixtures ───────────────────────────────────────────────


@pytest.fixture()
def mock_airsim():
    """Provide a mocked cosysairsim and the AirSimEnv.

    Yields:
        A tuple ``(env, client)`` where ``env`` is the ready-to-use
        ``AirSimEnv`` and ``client`` is the mock ``CarClient``.
    """
    mock_module, client = _make_mock_airsim_module()
    sys.modules["cosysairsim"] = mock_module

    bridge_module = importlib.import_module("airsim.communication.airsim_bridge")
    env_module = importlib.import_module("airsim.environments.airsim_env")

    reachability_sequence = [True, False, True, True]

    with patch.object(
        bridge_module,
        "is_ue5_reachable",
        side_effect=reachability_sequence,
    ), patch.object(bridge_module.time, "sleep", return_value=None), patch.object(env_module.time, "sleep", return_value=None):
        from airsim.environments.airsim_env import AirSimEnv

        env = AirSimEnv(
            ship_pos=[2245, -47],
            target_pos=[-1559, 11990],
            max_steps=100,
            use_pause_stepping=False,
            training_mode=False,
            verbose=False,
        )
        yield env, client

    if "cosysairsim" in sys.modules:
        del sys.modules["cosysairsim"]


class TestTrainingMode:
    """Verify the current rendering lifecycle used by AirSim training."""

    def test_training_mode_uses_shared_calibration_reference_overlay(self):
        """training_mode=True should reuse the same calibration overlay as visualization mode."""
        mock_module, client = _make_mock_airsim_module()
        overlay_calls = []

        def _fake_load_saved_calibration_reference(
            client_arg,
            calibration_path=None,
            draw_reference=True,
            ship_pos=None,
            target_pos=None,
        ):
            """Capture calibration-overlay arguments without touching UE5."""
            overlay_calls.append(
                {
                    "client": client_arg,
                    "calibration_path": calibration_path,
                    "draw_reference": draw_reference,
                    "ship_pos": ship_pos,
                    "target_pos": target_pos,
                }
            )
            return CoordTransform.identity((2245.0, -47.0)), {}

        module_name = "fw.common.airsim.tools.calibrate_coords"
        previous_module = sys.modules.get(module_name)
        sys.modules[module_name] = MagicMock(
            load_saved_calibration_reference=_fake_load_saved_calibration_reference
        )
        try:
            env = _make_env(mock_module, training_mode=True)
        finally:
            if previous_module is None:
                del sys.modules[module_name]
            else:
                sys.modules[module_name] = previous_module

        client.simRunConsoleCommand.assert_any_call("ShowFlag.Rendering 1")
        assert len(overlay_calls) == 1
        assert overlay_calls[0]["client"] is client
        assert overlay_calls[0]["draw_reference"] is True
        assert overlay_calls[0]["ship_pos"] == pytest.approx((2245.0, -47.0))
        assert overlay_calls[0]["target_pos"] == pytest.approx((-1559.0, 11990.0))
        env.close()

    def test_training_mode_can_disable_shared_calibration_reference_overlay(self):
        """Training mode should allow the shared calibration overlay to be disabled."""
        mock_module, _ = _make_mock_airsim_module()
        overlay_calls = []

        def _fake_load_saved_calibration_reference(*args, **kwargs):
            """Record overlay kwargs while returning an identity transform."""
            overlay_calls.append(kwargs)
            return CoordTransform.identity((2245.0, -47.0)), {}

        module_name = "fw.common.airsim.tools.calibrate_coords"
        previous_module = sys.modules.get(module_name)
        sys.modules[module_name] = MagicMock(
            load_saved_calibration_reference=_fake_load_saved_calibration_reference
        )
        try:
            env = _make_env(
                mock_module,
                training_mode=True,
                draw_calibration_reference=False,
            )
        finally:
            if previous_module is None:
                del sys.modules[module_name]
            else:
                sys.modules[module_name] = previous_module

        assert len(overlay_calls) == 1
        assert overlay_calls[0]["draw_reference"] is False
        env.close()

    def test_no_rendering_mode_disables_ue5_rendering(self):
        """no_rendering_mode=True should issue the UE5 rendering-off console command."""
        mock_module, client = _make_mock_airsim_module()
        env = _make_env(
            mock_module,
            training_mode=True,
            no_rendering_mode=True,
        )

        client.simRunConsoleCommand.assert_any_call("ShowFlag.Rendering 0")
        env.close()

    def test_close_restores_rendering(self):
        """Closing the env should always re-enable rendering for normal UE5 use."""
        mock_module, client = _make_mock_airsim_module()
        env = _make_env(mock_module, training_mode=True)
        client.simRunConsoleCommand.reset_mock()

        env.close()

        client.simRunConsoleCommand.assert_any_call("ShowFlag.Rendering 1")

    def test_step_draws_live_ship_debug_marker(self):
        """Each step should refresh the non-persistent live ship marker in UE5."""
        mock_module, client = _make_mock_airsim_module()
        env = _make_env(mock_module, training_mode=True)
        client.simPlotPoints.reset_mock()

        env.step(np.array([0.0, 0.0], dtype=np.float32))

        assert client.simPlotPoints.called
        assert client.simPlotPoints.call_args.kwargs["is_persistent"] is False


# ── T-009: Observation mapping & normalisation ─────────────


class TestObservationSpace:
    """T-009: Verify observation space shape, bounds, and normalisation."""

    def test_observation_space_shape(self, mock_airsim):
        """Observation space must be Box(12,) matching AirSimEnv."""
        env, _ = mock_airsim
        assert env.observation_space.shape == (12,)
        assert env.observation_space.dtype == np.float32

    def test_observation_bounds(self, mock_airsim):
        """Low/high arrays must match the documented channel ranges."""
        env, _ = mock_airsim
        expected_low = np.array(
            [-1, -1, -1, 0, 0, -2, -1, -1, -1, -1, -1, -1], dtype=np.float32
        )
        expected_high = np.array(
            [1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1], dtype=np.float32
        )
        np.testing.assert_array_equal(env.observation_space.low, expected_low)
        np.testing.assert_array_equal(env.observation_space.high, expected_high)

    def test_observation_within_bounds(self, mock_airsim):
        """Returned observation must always be clipped to obs-space."""
        env, _ = mock_airsim
        obs = env._get_obs()
        assert obs.shape == (12,)
        assert np.all(obs >= env.observation_space.low), (
            f"Obs below low bound: {obs}"
        )
        assert np.all(obs <= env.observation_space.high), (
            f"Obs above high bound: {obs}"
        )

    def test_normalization_saturation(self, mock_airsim):
        """T-009: Velocities beyond ship limits must saturate at ±1.0 (min-max clip).

        Sets surge (ship_u) to 10× the ship's maximum surge limit.
        The normalised channel 0 must clamp to 1.0, not exceed it.
        Channel 2 (yaw rate) is set to -10× limit and must clamp to -1.0.
        """
        env, _ = mock_airsim
        surge_max = env.ship_specs.surge_limits[1]
        yaw_max = env.ship_specs.yaw_rate_limits[1]

        env.ship_u = surge_max * 10.0   # way above max → should clip to 1.0
        env.ship_v = 0.0
        env.ship_r = -yaw_max * 10.0    # way below min → should clip to -1.0

        obs = env._get_obs()

        assert obs[0] == pytest.approx(1.0), (
            f"Saturated surge should normalise to 1.0, got {obs[0]}"
        )
        assert obs[2] == pytest.approx(-1.0), (
            f"Saturated yaw rate (neg) should normalise to -1.0, got {obs[2]}"
        )


# ── T-010: Adapter layer ──────────────────────────────────


class TestAdapterLayer:
    """T-010: Verify BaseEnv adapter compliance."""

    def test_type_name(self, mock_airsim):
        """``type_name`` property must return ``'AirSimEnv'``."""
        env, _ = mock_airsim
        assert env.type_name == "AirSimEnv"

    def test_env_specific_reset_exists(self, mock_airsim):
        """``env_specific_reset`` must be callable (overrides BaseEnv stub)."""
        env, _ = mock_airsim
        assert callable(getattr(env, "env_specific_reset", None))

    def test_randomize_exists(self, mock_airsim):
        """``randomize`` must be callable (overrides BaseEnv stub)."""
        env, _ = mock_airsim
        assert callable(getattr(env, "randomize", None))

    def test_randomize_rejects_negative_scale(self, mock_airsim):
        """``randomize`` must raise ValueError for non-positive scale."""
        env, _ = mock_airsim
        with pytest.raises(ValueError, match="positive"):
            env.randomize(randomization_scale=-1.0)

    def test_randomize_stays_anchored_to_spawn_reference(self, mock_airsim):
        """Repeated randomization should stay near the base spawn instead of drifting."""
        env, _ = mock_airsim
        base = env._base_initial_ship_pos.copy()
        rng = MagicMock()
        rng.uniform.return_value = np.array([10.0, -5.0], dtype=np.float32)

        with patch("airsim.environments.airsim_env.np.random.default_rng", return_value=rng):
            env.randomize(randomization_scale=12.0)
            first = env.initial_ship_pos.copy()
            env.randomize()
            second = env.initial_ship_pos.copy()

        expected = base + np.array([10.0, -5.0], dtype=np.float32)
        assert np.allclose(first, expected)
        assert np.allclose(second, expected)
        assert np.allclose(env.checkpoints[0]["pos"], expected)

    def test_curriculum_randomize_limits_active_route_window(self):
        """Curriculum mode should activate only the selected route slice."""
        mock_module, _ = _make_mock_airsim_module()
        env = _make_env(
            mock_module,
            curriculum_enabled=True,
            curriculum_window_size=5,
            curriculum_target_checkpoint=7,
        )

        env.randomize()

        assert len(env.checkpoints) - 1 == 5
        assert env._curriculum_active_end_index == 7
        assert env._curriculum_active_start_index == 2
        env.close()


# ── T-010: 2D (PySimEnv) vs 3D (AirSimEnv) observation parity ────────────


class TestObsSpaceParity:
    """T-010: Validate the current observation-space compatibility contract.

    AirSimEnv uses a flat 10-channel observation. PySimEnv currently exposes
    a flat 12-channel observation (the AirSim-compatible core plus 2 extra
    features).
    """

    @pytest.fixture(scope="class")
    def pysim_env(self):
        """Create a real PySimEnv (pure Python, no mocking required)."""
        from fw.environments.py_sim_env import PySimEnv
        env = PySimEnv(render_mode=None)
        yield env
        # no close needed – PySimEnv has no network connections

    def test_action_space_matches(self, mock_airsim, pysim_env):
        """Both environments must expose the same 2-D continuous action space."""
        airsim_env, _ = mock_airsim
        air_space = airsim_env.action_space
        py_space = pysim_env.action_space

        assert air_space.shape == py_space.shape, (
            f"Action space shape mismatch: AirSim={air_space.shape}, "
            f"PySimEnv={py_space.shape}"
        )
        np.testing.assert_array_equal(air_space.low, py_space.low,
                                      err_msg="action_space.low mismatch")
        np.testing.assert_array_equal(air_space.high, py_space.high,
                                      err_msg="action_space.high mismatch")

    def test_obs_channel_count_matches(self, mock_airsim, pysim_env):
        """PySimEnv must expose at least the AirSim 12-channel core."""
        airsim_env, _ = mock_airsim
        py_obs_shape = pysim_env.observation_space.shape
        air_obs_shape = airsim_env.observation_space.shape

        n_channels_pysim = py_obs_shape[-1]
        n_channels_airsim = air_obs_shape[0]

        assert n_channels_airsim == 12
        assert n_channels_pysim == 12
        assert n_channels_pysim >= n_channels_airsim

    def test_obs_bounds_per_channel_match(self, mock_airsim, pysim_env):
        """Validate matching bounds for shared channels and known differences."""
        airsim_env, _ = mock_airsim
        py_low = np.asarray(pysim_env.observation_space.low, dtype=np.float32).reshape(-1)
        py_high = np.asarray(pysim_env.observation_space.high, dtype=np.float32).reshape(-1)

        air_low = airsim_env.observation_space.low
        air_high = airsim_env.observation_space.high

        # Shared channels that should remain aligned:
        # 0..2 = velocities, 6..9 = heading errors.
        shared_indices = [0, 1, 2, 6, 7, 8, 9]
        np.testing.assert_array_equal(
            py_low[shared_indices],
            air_low[shared_indices],
            err_msg="Shared-channel low bounds mismatch",
        )
        np.testing.assert_array_equal(
            py_high[shared_indices],
            air_high[shared_indices],
            err_msg="Shared-channel high bounds mismatch",
        )

        # Explicitly document the intentional range differences:
        # 3..4 are distances, 5 is signed cross-track error.
        np.testing.assert_array_equal(air_low[3:6], np.array([0.0, 0.0, -2.0], dtype=np.float32))
        np.testing.assert_array_equal(air_high[3:6], np.array([1.0, 1.0, 2.0], dtype=np.float32))

    def test_obs_shape_difference_is_documented(self, mock_airsim, pysim_env):
        """PySimEnv and AirSimEnv both expose flat observations.

        Callers that transfer policies between environments must handle this
        channel-count difference explicitly.
        """
        airsim_env, _ = mock_airsim
        assert airsim_env.observation_space.shape == (12,)
        assert pysim_env.observation_space.shape == (12,)




class TestWaypoints3D:
    """T-011: Verify 3D AirSim coordinates are attached to checkpoints."""

    def test_checkpoints_have_3d_coords(self, mock_airsim):
        """Every checkpoint must contain ``pos_airsim_3d`` (shape 3)."""
        env, _ = mock_airsim
        assert len(env.checkpoints) > 0
        for cp in env.checkpoints:
            assert "pos_airsim_3d" in cp, "Missing 3D transform"
            assert cp["pos_airsim_3d"].shape == (3,)
            assert cp["pos_airsim_3d"].dtype == np.float32

    def test_2d_and_3d_coords_consistent(self, mock_airsim):
        """The 2D Fossen pos and 3D AirSim pos must be related by
        the coordinate transform (round-trip check)."""
        env, _ = mock_airsim
        cp = env.checkpoints[0]
        fossen_x, fossen_y = cp["pos"]
        ax, ay, _ = env.coord.fossen_to_airsim(float(fossen_x), float(fossen_y))
        np.testing.assert_allclose(
            cp["pos_airsim_3d"][:2],
            [ax, ay],
            atol=0.01,
            err_msg="2D/3D coordinate mismatch",
        )


# ── T-012: Cross-track error in step info ─────────────────


class TestCrossTrackCSV:
    """T-012: Verify cross-track error is exposed in step info."""

    def test_step_info_contains_cross_track(self, mock_airsim):
        """``step()`` info dict must include ``cross_track_error``."""
        env, _ = mock_airsim
        action = np.array([0.0, 0.0], dtype=np.float32)
        _, _, _, _, info = env.step(action)
        assert "cross_track_error" in info
        assert isinstance(info["cross_track_error"], float)

    def test_step_info_contains_checkpoint_index(self, mock_airsim):
        """``step()`` info dict must include ``checkpoint_index``."""
        env, _ = mock_airsim
        action = np.array([0.0, 0.0], dtype=np.float32)
        _, _, _, _, info = env.step(action)
        assert "checkpoint_index" in info
        assert isinstance(info["checkpoint_index"], int)
        assert "reached_checkpoint_count" in info
        assert "total_checkpoints" in info
        assert "checkpoint_fraction" in info
        assert isinstance(info["reached_checkpoint_count"], int)
        assert isinstance(info["total_checkpoints"], int)
        assert isinstance(info["checkpoint_fraction"], float)

    def test_step_info_contains_latency_fields(self, mock_airsim):
        """Lock-step telemetry must expose latency and UE5 timestamp info."""
        env, _ = mock_airsim
        action = np.array([0.0, 0.0], dtype=np.float32)
        _, _, _, _, info = env.step(action)
        assert "physics_wait_s" in info
        assert "step_latency_s" in info
        assert "ue5_timestamp" in info
        assert isinstance(info["physics_wait_s"], float)
        assert isinstance(info["step_latency_s"], float)
        assert isinstance(info["ue5_timestamp"], int)

    def test_step_info_contains_stuck_tracking_fields(self, mock_airsim):
        """Step info should expose the live stuck-detection counters."""
        env, _ = mock_airsim
        action = np.array([0.0, 0.0], dtype=np.float32)
        _, _, _, _, info = env.step(action)
        assert "raw_progress" in info
        assert "low_speed_steps" in info
        assert "stalled_progress_steps" in info
        assert "high_cross_track_steps" in info
        assert "hard_stop_steps" in info
        assert "obstacle_contact_steps" in info
        assert "checkpoint_stagnation_steps" in info
        assert isinstance(info["raw_progress"], float)
        assert isinstance(info["low_speed_steps"], int)
        assert isinstance(info["stalled_progress_steps"], int)
        assert isinstance(info["high_cross_track_steps"], int)
        assert isinstance(info["hard_stop_steps"], int)
        assert isinstance(info["obstacle_contact_steps"], int)
        assert isinstance(info["checkpoint_stagnation_steps"], int)

    def test_non_pause_step_scales_sleep_with_simulation_speed(self):
        """Non-pause stepping should sleep in wall-clock time scaled by simulation speed."""
        mock_module, _ = _make_mock_airsim_module()
        env = _make_env(
            mock_module,
            use_pause_stepping=False,
            simulation_speed=12.0,
            time_step=0.24,
        )
        env_module = importlib.import_module("airsim.environments.airsim_env")
        action = np.array([0.0, 0.0], dtype=np.float32)

        with patch.object(env_module.time, "sleep", return_value=None) as sleep_mock:
            env.step(action)

        sleep_mock.assert_any_call(pytest.approx(0.02))

    def test_non_pause_step_explicitly_unpauses_simulation(self):
        """Continuous training mode must unpause UE5 before applying controls."""
        mock_module, _ = _make_mock_airsim_module()
        env = _make_env(
            mock_module,
            use_pause_stepping=False,
            simulation_speed=30.0,
        )
        env.bridge.set_pause = MagicMock()

        env.step(np.array([0.0, 0.0], dtype=np.float32))

        env.bridge.set_pause.assert_any_call(False)

    def test_pause_step_can_advance_by_simulation_time(self):
        """Paused training can use sim-time stepping instead of frame stepping."""
        mock_module, _ = _make_mock_airsim_module()
        env = _make_env(
            mock_module,
            use_pause_stepping=True,
            use_sim_time_stepping=True,
            time_step=0.1,
        )
        env.bridge.continue_for_time = MagicMock()
        env.bridge.is_paused = MagicMock(side_effect=[False, True])

        env.step(np.array([0.0, 0.0], dtype=np.float32))

        env.bridge.continue_for_time.assert_called_once_with(0.1)

    def test_turn_action_governor_boosts_steering_and_reduces_throttle_on_sharp_turn(self, mock_airsim):
        """Sharp turns should trigger the AirSim action governor."""
        env, _ = mock_airsim
        checkpoint_index = _find_turn_checkpoint(env)
        if checkpoint_index is None:
            pytest.skip("No turn checkpoint available in route")

        prev_pos = env.checkpoints[checkpoint_index - 1]["pos"]
        cur_pos = env.checkpoints[checkpoint_index]["pos"]
        path_vec = cur_pos - prev_pos
        path_len = np.linalg.norm(path_vec)
        unit = path_vec / path_len
        path_heading = float(np.arctan2(path_vec[1], path_vec[0]))
        near_checkpoint_pos = prev_pos + unit * (0.85 * path_len)

        env.checkpoint_index = checkpoint_index
        env.previous_ship_pos = (near_checkpoint_pos - unit * 1.0).astype(np.float32)
        env.ship_pos = near_checkpoint_pos.astype(np.float32)
        env.previous_heading = path_heading + np.deg2rad(30.0)
        env.ship_heading = path_heading + np.deg2rad(30.0)
        env.ship_speed = 5.0

        raw_action = np.array([0.10, 0.80], dtype=np.float32)
        governed_action = env._apply_turn_action_governor(raw_action)

        assert abs(float(governed_action[0])) > abs(float(raw_action[0]))
        assert float(governed_action[1]) < float(raw_action[1])

    def test_turn_action_governor_injects_steering_for_timid_policy(self, mock_airsim):
        """The governor should add turn commitment even when the policy outputs zero steering."""
        env, _ = mock_airsim
        checkpoint_index = _find_turn_checkpoint(env)
        if checkpoint_index is None:
            pytest.skip("No turn checkpoint available in route")

        prev_pos = env.checkpoints[checkpoint_index - 1]["pos"]
        cur_pos = env.checkpoints[checkpoint_index]["pos"]
        path_vec = cur_pos - prev_pos
        path_len = np.linalg.norm(path_vec)
        unit = path_vec / path_len
        path_heading = float(np.arctan2(path_vec[1], path_vec[0]))
        near_checkpoint_pos = prev_pos + unit * (0.88 * path_len)

        env.checkpoint_index = checkpoint_index
        env.previous_ship_pos = (near_checkpoint_pos - unit * 1.0).astype(np.float32)
        env.ship_pos = near_checkpoint_pos.astype(np.float32)
        env.previous_heading = path_heading + np.deg2rad(28.0)
        env.ship_heading = path_heading + np.deg2rad(28.0)
        env.previous_action = np.array([0.0, 0.70], dtype=np.float32)
        env.ship_speed = 5.0

        raw_action = np.array([0.0, 0.80], dtype=np.float32)
        governed_action = env._apply_turn_action_governor(raw_action)

        assert abs(float(governed_action[0])) > 0.05
        assert float(governed_action[1]) < float(raw_action[1])

    def test_turn_action_governor_leaves_straight_action_unchanged(self, mock_airsim):
        """Straight segments should not be distorted by the turn governor."""
        env, _ = mock_airsim
        checkpoint_index = _find_straight_checkpoint(env)
        if checkpoint_index is None:
            pytest.skip("No sufficiently straight checkpoint available in route")
        _set_reward_state(
            env,
            progress_m=12.0,
            offset_m=0.0,
            speed_mps=3.0,
            checkpoint_index=checkpoint_index,
            heading_offset_rad=0.0,
            steering=0.0,
            throttle=0.0,
        )

        raw_action = np.array([0.10, 0.60], dtype=np.float32)
        governed_action = env._apply_turn_action_governor(raw_action)

        np.testing.assert_allclose(governed_action, raw_action, atol=1e-6)

    def test_reward_penalizes_large_rudder_change(self, mock_airsim):
        """Large rudder changes should reduce reward relative to smooth steering."""
        env, _ = mock_airsim
        _set_reward_state(env, progress_m=10.0, offset_m=0.0, speed_mps=1.0)

        env.previous_action = np.array([0.0, 0.5], dtype=np.float32)
        env.performed_action = np.array([0.0, 0.5], dtype=np.float32)
        smooth_reward, _, _ = env._calculate_reward()

        env.previous_action = np.array([0.0, 0.5], dtype=np.float32)
        env.performed_action = np.array([env.ship_specs.max_rudder_angle, 0.5], dtype=np.float32)
        flap_reward, _, _ = env._calculate_reward()

        assert flap_reward < smooth_reward

    def test_reward_penalizes_no_forward_progress_more_strongly(self, mock_airsim):
        """A stalled step should score worse than a forward-moving step."""
        env, _ = mock_airsim
        _set_reward_state(env, progress_m=10.0, offset_m=0.0, speed_mps=1.0)
        forward_reward, _, _ = env._calculate_reward()

        env._low_speed_steps = 0
        env._stalled_progress_steps = 0
        env._high_cross_track_steps = 0
        _set_reward_state(env, progress_m=0.0, offset_m=0.0, speed_mps=1.0)
        stalled_reward, _, _ = env._calculate_reward()

        assert stalled_reward < forward_reward

    def test_late_checkpoints_penalize_cross_track_more_strongly(self, mock_airsim):
        """The same lateral error should hurt more after checkpoint 3."""
        env, _ = mock_airsim
        _set_reward_state(env, progress_m=10.0, offset_m=80.0, speed_mps=1.0, checkpoint_index=2)
        early_reward, _, _ = env._calculate_reward()

        env._low_speed_steps = 0
        env._stalled_progress_steps = 0
        env._high_cross_track_steps = 0
        _set_reward_state(env, progress_m=10.0, offset_m=80.0, speed_mps=1.0, checkpoint_index=4)
        late_reward, _, _ = env._calculate_reward()

        assert late_reward < early_reward

    def test_offtrack_recovery_heading_is_better_than_heading_further_away(self, mock_airsim):
        """When far off track, steering back toward the route should score better."""
        env, _ = mock_airsim
        checkpoint_index = 4
        prev_pos = env.checkpoints[checkpoint_index - 1]["pos"]
        cur_pos = env.checkpoints[checkpoint_index]["pos"]
        path_vec = cur_pos - prev_pos
        path_len = np.linalg.norm(path_vec)
        path_unit = path_vec / path_len
        path_heading = float(np.arctan2(path_vec[1], path_vec[0]))
        normal = np.array([-path_unit[1], path_unit[0]], dtype=np.float32)

        env.checkpoint_index = checkpoint_index
        env.ship_speed = 1.0
        env.ship_r = 0.0
        env.previous_action = np.array([0.0, 0.4], dtype=np.float32)
        env.performed_action = np.array([0.0, 0.4], dtype=np.float32)
        env.last_controls = {"steering": 0.0, "throttle": 0.4, "brake": 0.0}
        env._ship_in_open_water = lambda pos: True
        env._check_collision = lambda: False

        base_pos = prev_pos + path_unit * (0.55 * path_len)
        ship_pos = base_pos + normal * 120.0
        previous_ship_pos = ship_pos - path_unit * 1.0

        env.previous_ship_pos = previous_ship_pos.astype(np.float32)
        env.ship_pos = ship_pos.astype(np.float32)
        env.previous_heading = path_heading - np.deg2rad(35.0)
        env.ship_heading = path_heading - np.deg2rad(35.0)
        recovery_reward, _, _ = env._calculate_reward()

        env._low_speed_steps = 0
        env._stalled_progress_steps = 0
        env._high_cross_track_steps = 0
        env._hard_stop_steps = 0
        env._best_checkpoint_distance = float("inf")
        env._checkpoint_stagnation_steps = 0
        env.previous_ship_pos = previous_ship_pos.astype(np.float32)
        env.ship_pos = ship_pos.astype(np.float32)
        env.previous_heading = path_heading + np.deg2rad(35.0)
        env.ship_heading = path_heading + np.deg2rad(35.0)
        diverging_reward, _, _ = env._calculate_reward()

        assert recovery_reward > diverging_reward

    def test_heading_error_with_high_throttle_is_penalized(self, mock_airsim):
        """High throttle while misaligned should score worse than gentle throttle."""
        env, _ = mock_airsim
        heading_offset = np.deg2rad(55.0)

        _set_reward_state(
            env,
            progress_m=10.0,
            offset_m=0.0,
            speed_mps=1.0,
            heading_offset_rad=heading_offset,
            throttle=0.9,
        )
        high_throttle_reward, _, _ = env._calculate_reward()

        env._low_speed_steps = 0
        env._stalled_progress_steps = 0
        env._high_cross_track_steps = 0
        _set_reward_state(
            env,
            progress_m=10.0,
            offset_m=0.0,
            speed_mps=1.0,
            heading_offset_rad=heading_offset,
            throttle=0.2,
        )
        low_throttle_reward, _, _ = env._calculate_reward()

        assert high_throttle_reward < low_throttle_reward

    def test_high_throttle_while_far_offtrack_is_penalized_more(self, mock_airsim):
        """High throttle should score worse than gentle throttle when already far off-track."""
        env, _ = mock_airsim

        _set_reward_state(
            env,
            progress_m=10.0,
            offset_m=140.0,
            speed_mps=env.MAX_SPEED,
            throttle=0.9,
        )
        high_throttle_reward, _, _ = env._calculate_reward()

        env._low_speed_steps = 0
        env._stalled_progress_steps = 0
        env._high_cross_track_steps = 0
        env._hard_stop_steps = 0
        env._obstacle_contact_steps = 0
        env._best_checkpoint_distance = float("inf")
        env._checkpoint_stagnation_steps = 0
        _set_reward_state(
            env,
            progress_m=10.0,
            offset_m=140.0,
            speed_mps=env.MAX_SPEED,
            throttle=0.2,
        )
        low_throttle_reward, _, _ = env._calculate_reward()

        assert high_throttle_reward < low_throttle_reward

    def test_understeering_while_offtrack_scores_worse_than_committing_to_turn(self, mock_airsim):
        """When far off-route and misaligned, timid steering should score worse than a corrective turn."""
        env, _ = mock_airsim
        _set_reward_state(
            env,
            progress_m=8.0,
            offset_m=120.0,
            speed_mps=1.0,
            checkpoint_index=4,
            heading_offset_rad=np.deg2rad(35.0),
            steering=0.0,
            throttle=0.4,
        )
        timid_reward, _, _ = env._calculate_reward()

        env._low_speed_steps = 0
        env._stalled_progress_steps = 0
        env._high_cross_track_steps = 0
        env._best_checkpoint_distance = float("inf")
        env._checkpoint_stagnation_steps = 0
        _set_reward_state(
            env,
            progress_m=8.0,
            offset_m=120.0,
            speed_mps=1.0,
            checkpoint_index=4,
            heading_offset_rad=np.deg2rad(35.0),
            steering=0.18,
            throttle=0.4,
        )
        committed_reward, _, _ = env._calculate_reward()

        assert timid_reward < committed_reward

    def test_reducing_heading_error_scores_better_than_worsening_it(self, mock_airsim):
        """Reward should prefer steps that reduce heading error over steps that increase it."""
        env, _ = mock_airsim
        checkpoint_index = 4
        _set_reward_state(
            env,
            progress_m=8.0,
            offset_m=40.0,
            speed_mps=1.0,
            checkpoint_index=checkpoint_index,
            heading_offset_rad=np.deg2rad(15.0),
            steering=0.12,
            throttle=0.4,
        )
        prev_pos = env.checkpoints[checkpoint_index - 1]["pos"]
        cur_pos = env.checkpoints[checkpoint_index]["pos"]
        path_heading = float(np.arctan2(*(cur_pos - prev_pos)[::-1]))
        env.previous_heading = path_heading + np.deg2rad(35.0)
        env.ship_heading = path_heading + np.deg2rad(15.0)
        improving_reward, _, _ = env._calculate_reward()

        env._low_speed_steps = 0
        env._stalled_progress_steps = 0
        env._high_cross_track_steps = 0
        env._best_checkpoint_distance = float("inf")
        env._checkpoint_stagnation_steps = 0
        _set_reward_state(
            env,
            progress_m=8.0,
            offset_m=40.0,
            speed_mps=1.0,
            checkpoint_index=checkpoint_index,
            heading_offset_rad=np.deg2rad(35.0),
            steering=0.12,
            throttle=0.4,
        )
        env.previous_heading = path_heading + np.deg2rad(15.0)
        env.ship_heading = path_heading + np.deg2rad(35.0)
        worsening_reward, _, _ = env._calculate_reward()

        assert improving_reward > worsening_reward

    def test_high_speed_in_turn_is_penalized_more_than_slowing_down(self, mock_airsim):
        """Sharp-turn approach should prefer slowing down over staying at top speed."""
        env, _ = mock_airsim
        checkpoint_index = _find_turn_checkpoint(env)
        if checkpoint_index is None:
            pytest.skip("No turn checkpoint available in route")

        prev_pos = env.checkpoints[checkpoint_index - 1]["pos"]
        cur_pos = env.checkpoints[checkpoint_index]["pos"]
        path_vec = cur_pos - prev_pos
        path_len = np.linalg.norm(path_vec)
        unit = path_vec / path_len
        near_checkpoint_pos = prev_pos + unit * (0.85 * path_len)
        heading_offset = np.deg2rad(30.0)

        env.checkpoint_index = checkpoint_index
        env.previous_ship_pos = (near_checkpoint_pos - unit * 1.0).astype(np.float32)
        env.ship_pos = near_checkpoint_pos.astype(np.float32)
        env.previous_action = np.array([0.0, 0.4], dtype=np.float32)
        env.performed_action = np.array([0.08, 0.4], dtype=np.float32)
        env.last_controls = {"steering": 0.08, "throttle": 0.4, "brake": 0.0}
        env._ship_in_open_water = lambda pos: True
        env._check_collision = lambda: False

        path_heading = float(np.arctan2(path_vec[1], path_vec[0]))
        env.previous_heading = path_heading + heading_offset
        env.ship_heading = path_heading + heading_offset

        env.ship_speed = 5.0
        env.ship_r = 0.0
        fast_reward, _, _ = env._calculate_reward()

        env._low_speed_steps = 0
        env._stalled_progress_steps = 0
        env._high_cross_track_steps = 0
        env._best_checkpoint_distance = float("inf")
        env._checkpoint_stagnation_steps = 0
        env.ship_speed = 2.5
        slow_reward, _, _ = env._calculate_reward()

        assert fast_reward < slow_reward

    def test_sharp_turn_prefers_committed_steering_over_timid_steering(self, mock_airsim):
        """A tight turn should score better with committed steering than with timid input."""
        env, _ = mock_airsim
        checkpoint_index = _find_turn_checkpoint(env)
        if checkpoint_index is None:
            pytest.skip("No turn checkpoint available in route")

        prev_pos = env.checkpoints[checkpoint_index - 1]["pos"]
        cur_pos = env.checkpoints[checkpoint_index]["pos"]
        path_vec = cur_pos - prev_pos
        path_len = np.linalg.norm(path_vec)
        unit = path_vec / path_len
        near_checkpoint_pos = prev_pos + unit * (0.82 * path_len)
        path_heading = float(np.arctan2(path_vec[1], path_vec[0]))
        heading_offset = np.deg2rad(28.0)

        env.checkpoint_index = checkpoint_index
        env.previous_ship_pos = (near_checkpoint_pos - unit * 1.0).astype(np.float32)
        env.ship_pos = near_checkpoint_pos.astype(np.float32)
        env.previous_heading = path_heading + heading_offset
        env.ship_heading = path_heading + heading_offset
        env.ship_speed = 3.2
        env.ship_r = 0.0
        env.previous_action = np.array([0.0, 0.45], dtype=np.float32)
        env._ship_in_open_water = lambda pos: True
        env._check_collision = lambda: False

        env.performed_action = np.array([0.04, 0.45], dtype=np.float32)
        env.last_controls = {"steering": 0.04, "throttle": 0.45, "brake": 0.0}
        timid_reward, _, _ = env._calculate_reward()

        env._low_speed_steps = 0
        env._stalled_progress_steps = 0
        env._high_cross_track_steps = 0
        env._hard_stop_steps = 0
        env._obstacle_contact_steps = 0
        env._best_checkpoint_distance = float("inf")
        env._checkpoint_stagnation_steps = 0
        env.previous_ship_pos = (near_checkpoint_pos - unit * 1.0).astype(np.float32)
        env.ship_pos = near_checkpoint_pos.astype(np.float32)
        env.previous_heading = path_heading + heading_offset
        env.ship_heading = path_heading + heading_offset
        env.ship_speed = 3.2
        env.ship_r = 0.0
        env.previous_action = np.array([0.0, 0.45], dtype=np.float32)
        env.performed_action = np.array([0.28, 0.45], dtype=np.float32)
        env.last_controls = {"steering": 0.28, "throttle": 0.45, "brake": 0.0}
        committed_reward, _, _ = env._calculate_reward()

        assert committed_reward > timid_reward

    def test_straight_segment_still_prefers_smoother_steering(self, mock_airsim):
        """Needless hard steering on a straight segment should still score worse."""
        env, _ = mock_airsim
        _set_reward_state(
            env,
            progress_m=12.0,
            offset_m=0.0,
            speed_mps=3.0,
            checkpoint_index=1,
            heading_offset_rad=0.0,
            steering=0.04,
            throttle=0.4,
        )
        env.previous_action = np.array([0.0, 0.4], dtype=np.float32)
        env.last_controls = {"steering": 0.04, "throttle": 0.4, "brake": 0.0}
        smooth_reward, _, _ = env._calculate_reward()

        env._low_speed_steps = 0
        env._stalled_progress_steps = 0
        env._high_cross_track_steps = 0
        env._hard_stop_steps = 0
        env._obstacle_contact_steps = 0
        env._best_checkpoint_distance = float("inf")
        env._checkpoint_stagnation_steps = 0
        _set_reward_state(
            env,
            progress_m=12.0,
            offset_m=0.0,
            speed_mps=3.0,
            checkpoint_index=1,
            heading_offset_rad=0.0,
            steering=0.28,
            throttle=0.4,
        )
        env.previous_action = np.array([0.0, 0.4], dtype=np.float32)
        env.last_controls = {"steering": 0.28, "throttle": 0.4, "brake": 0.0}
        hard_turn_reward, _, _ = env._calculate_reward()

        assert hard_turn_reward < smooth_reward

    def test_high_yaw_rate_and_steering_change_are_penalized_more(self, mock_airsim):
        """Aggressive steering dynamics should reduce reward."""
        env, _ = mock_airsim
        _set_reward_state(
            env,
            progress_m=10.0,
            offset_m=0.0,
            speed_mps=1.0,
            steering=0.0,
            throttle=0.5,
            yaw_rate=0.0,
        )
        env.previous_action = np.array([0.0, 0.5], dtype=np.float32)
        smooth_reward, _, _ = env._calculate_reward()

        env._low_speed_steps = 0
        env._stalled_progress_steps = 0
        env._high_cross_track_steps = 0
        _set_reward_state(
            env,
            progress_m=10.0,
            offset_m=0.0,
            speed_mps=1.0,
            steering=env.ship_specs.max_rudder_angle,
            throttle=0.5,
            yaw_rate=env.ship_specs.yaw_rate_limits[1],
        )
        env.previous_action = np.array([0.0, 0.5], dtype=np.float32)
        aggressive_reward, _, _ = env._calculate_reward()

        assert aggressive_reward < smooth_reward

    def test_checkpoint_entry_receives_large_reward_bonus(self, mock_airsim):
        """Entering a checkpoint should grant a large discrete reward bonus."""
        env, _ = mock_airsim
        _set_reward_state(env, progress_m=10.0, offset_m=0.0, speed_mps=1.0, checkpoint_index=2)
        baseline_reward, _, _ = env._calculate_reward()

        env._low_speed_steps = 0
        env._stalled_progress_steps = 0
        env._high_cross_track_steps = 0
        current_cp = env.checkpoints[2]
        _set_reward_state(
            env,
            progress_m=0.0,
            offset_m=0.0,
            speed_mps=1.0,
            checkpoint_index=2,
        )
        env.ship_pos = current_cp["pos"].copy()
        env.previous_ship_pos = current_cp["pos"].copy()
        checkpoint_reward, _, _ = env._calculate_reward()

        assert checkpoint_reward > baseline_reward + 20.0
        assert env.checkpoint_index == 3

    def test_high_cross_track_time_penalty_grows_when_agent_stays_off_track(self, mock_airsim):
        """Persistent large cross-track error should worsen reward over time."""
        env, _ = mock_airsim
        _set_reward_state(env, progress_m=10.0, offset_m=220.0, speed_mps=1.0)
        reward_1, done_1, _ = env._calculate_reward()

        _set_reward_state(env, progress_m=10.0, offset_m=220.0, speed_mps=1.0)
        reward_2, done_2, _ = env._calculate_reward()

        assert done_1 is False
        assert done_2 is False
        assert reward_2 < reward_1

    def test_near_checkpoint_reward_softens_penalty_for_upcoming_turn_heading(self, mock_airsim):
        """Near a turning checkpoint, the next-segment heading should become less bad."""
        env, _ = mock_airsim
        checkpoint_index = _find_turn_checkpoint(env)
        if checkpoint_index is None:
            pytest.skip("Trajectory does not contain a sufficiently sharp turn for lookahead testing.")

        prev_pos = env.checkpoints[checkpoint_index - 1]["pos"]
        cur_pos = env.checkpoints[checkpoint_index]["pos"]
        next_pos = env.checkpoints[checkpoint_index + 1]["pos"]
        path_vec = cur_pos - prev_pos
        path_len = np.linalg.norm(path_vec)
        unit = path_vec / path_len
        far_pos = prev_pos + unit * (0.25 * path_len)
        near_checkpoint_pos = prev_pos + unit * (0.88 * path_len)
        next_heading = float(np.arctan2((next_pos - cur_pos)[1], (next_pos - cur_pos)[0]))

        env.checkpoint_index = checkpoint_index
        env.ship_speed = 1.0
        env.ship_r = 0.0
        env.previous_action = np.array([0.0, 0.4], dtype=np.float32)
        env.performed_action = np.array([0.0, 0.4], dtype=np.float32)
        env.last_controls = {"steering": 0.0, "throttle": 0.4, "brake": 0.0}
        env._ship_in_open_water = lambda pos: True
        env._check_collision = lambda: False

        env.previous_ship_pos = (far_pos - unit * 1.0).astype(np.float32)
        env.ship_pos = far_pos.astype(np.float32)
        env.previous_heading = next_heading
        env.ship_heading = next_heading
        far_reward, _, _ = env._calculate_reward()

        env._low_speed_steps = 0
        env._stalled_progress_steps = 0
        env._high_cross_track_steps = 0
        env.previous_ship_pos = (near_checkpoint_pos - unit * 1.0).astype(np.float32)
        env.ship_pos = near_checkpoint_pos.astype(np.float32)
        env.previous_heading = next_heading
        env.ship_heading = next_heading
        near_reward, _, _ = env._calculate_reward()

        assert near_reward > far_reward

    def test_collision_terminates_with_stronger_penalty_and_halts_vehicle(self, mock_airsim):
        """A collision should immediately end the episode and command a stop."""
        env, _ = mock_airsim
        _set_reward_state(env, progress_m=10.0, offset_m=0.0, speed_mps=1.0)
        env._check_collision = lambda: True
        env.bridge.set_controls = MagicMock()

        reward, done, reason = env._calculate_reward()

        assert done is True
        assert reward == -400.0
        assert reason == "collision"
        assert env.bridge.set_controls.called

    def test_collision_like_stop_terminates_and_punishes(self, mock_airsim):
        """A brief hard stop under throttle should be treated like a collision."""
        env, _ = mock_airsim
        env_module = importlib.import_module("airsim.environments.airsim_env")
        _set_reward_state(env, progress_m=0.0, offset_m=30.0, speed_mps=0.0)
        env.previous_speed = 2.5
        env.last_controls = {"steering": 0.0, "throttle": 0.8, "brake": 0.0}
        env._hard_stop_steps = env_module.COLLISION_LIKE_STOP_PATIENCE_STEPS - 1
        env._check_collision = lambda: False
        env.bridge.set_controls = MagicMock()

        reward, done, reason = env._calculate_reward()

        assert done is True
        assert reward == -400.0
        assert reason == "collision-like-stop"
        assert env.bridge.set_controls.called

    def test_obstacle_contact_stall_terminates_and_punishes(self, mock_airsim):
        """Persistent blocked-against-wall behavior should reset even without a collision flag."""
        env, _ = mock_airsim
        env_module = importlib.import_module("airsim.environments.airsim_env")
        _set_reward_state(env, progress_m=0.0, offset_m=80.0, speed_mps=0.2)
        env.previous_speed = 1.8
        env.last_controls = {"steering": 0.6, "throttle": 0.8, "brake": 0.0}
        env._obstacle_contact_steps = env_module.OBSTACLE_CONTACT_PATIENCE_STEPS - 1
        env._check_collision = lambda: False
        env.bridge.set_controls = MagicMock()

        reward, done, reason = env._calculate_reward()

        assert done is True
        assert reward == -400.0
        assert reason == "obstacle-contact-stall"
        assert env.bridge.set_controls.called

    def test_terminates_when_checkpoint_stagnates_while_still_far(self, mock_airsim):
        """A checkpoint should reset if the agent stops improving while still far away."""
        env, _ = mock_airsim
        env_module = importlib.import_module("airsim.environments.airsim_env")
        _set_reward_state(env, progress_m=1.0, offset_m=160.0, speed_mps=1.0)
        current_cp = env.checkpoints[env.checkpoint_index]["pos"]
        env._best_checkpoint_distance = float(np.linalg.norm(env.ship_pos - current_cp))
        env._checkpoint_stagnation_steps = env_module.CHECKPOINT_STAGNATION_PATIENCE_STEPS
        env._check_collision = lambda: False

        reward, done, reason = env._calculate_reward()

        assert done is True
        assert reward == -250
        assert reason == "checkpoint-stagnation"

    def test_terminates_when_speed_stays_too_low(self, mock_airsim):
        """Repeated near-zero speed should trigger an early stuck termination."""
        env, _ = mock_airsim
        env_module = importlib.import_module("airsim.environments.airsim_env")
        _set_reward_state(env, progress_m=0.5, offset_m=0.0, speed_mps=0.0)
        env._low_speed_steps = env_module.LOW_SPEED_PATIENCE_STEPS - 1

        reward, done, reason = env._calculate_reward()

        assert done is True
        assert reward == -250
        assert reason == "low-speed-stall"

    def test_terminates_when_progress_stalls_for_too_long(self, mock_airsim):
        """Repeated no-progress steps should terminate the episode early."""
        env, _ = mock_airsim
        env_module = importlib.import_module("airsim.environments.airsim_env")
        _set_reward_state(env, progress_m=0.0, offset_m=0.0, speed_mps=1.0)
        env._stalled_progress_steps = env_module.NO_PROGRESS_PATIENCE_STEPS - 1

        reward, done, reason = env._calculate_reward()

        assert done is True
        assert reward == -250
        assert reason == "no-progress-stall"

    def test_terminates_earlier_for_large_cross_track_error(self, mock_airsim):
        """Cross-track error above the tuned early-stop threshold should terminate."""
        env, _ = mock_airsim
        _set_reward_state(env, progress_m=10.0, offset_m=410.0, speed_mps=1.0)

        reward, done, reason = env._calculate_reward()

        assert done is True
        assert reward == -250
        assert reason == "cross-track"

    def test_episode_csv_logging(self, mock_airsim, tmp_path):
        """T-012: step() must write one telemetry row to the episode CSV.

        Verifies that when ``episode_log_path`` is configured on the env,
        each call to ``step()`` appends a row containing at minimum:
        ``episode``, ``episode_step``, ``cross_track_error``,
        ``checkpoint_index``, ``reward``, ``steering``, and ``throttle``.
        """
        env, _ = mock_airsim

        log_file = tmp_path / "episode_telemetry.csv"
        env._episode_log_path = str(log_file)
        env._open_episode_log()

        action = np.array([0.2, 0.6], dtype=np.float32)
        env.step(action)
        env.step(action)

        # Flush and close so the file contents are visible
        env._close_episode_log()

        assert log_file.exists(), "Episode CSV was not created"

        with open(log_file, newline="") as f:
            rows = list(csv.DictReader(f))

        assert len(rows) == 2, f"Expected 2 data rows, got {len(rows)}"

        required_columns = {
            "episode", "episode_step",
            "ship_x", "ship_y", "heading_deg", "speed",
            "cross_track_error", "checkpoint_index",
            "reward", "steering", "throttle",
            "physics_wait_s", "step_latency_s", "ue5_timestamp",
        }
        assert required_columns.issubset(rows[0].keys()), (
            f"Missing columns: {required_columns - set(rows[0].keys())}"
        )

        # Cross-track must be a parseable float >= 0
        cte = float(rows[0]["cross_track_error"])
        assert cte >= 0.0, f"cross_track_error must be non-negative, got {cte}"

    def test_headless_progress_heartbeat_prints(self, capsys):
        """Headless training should emit a console heartbeat even with rendering off."""
        mock_module, _ = _make_mock_airsim_module()
        env = _make_env(
            mock_module,
            training_mode=True,
            no_rendering_mode=True,
            progress_log_interval_steps=1,
        )

        env.step(np.array([0.0, 0.0], dtype=np.float32))

        captured = capsys.readouterr()
        assert "[AirSimEnv][progress]" in captured.out

    def test_pause_watchdog_warning_can_be_suppressed(self, mock_airsim):
        """Pause watchdog timeouts should stay out of the console by default."""
        env, _ = mock_airsim
        env.use_pause_stepping = True
        env._physics_update_timeout = 0.0
        env._log_pause_watchdog_warnings = False
        env.bridge.is_paused = MagicMock(return_value=False)
        action = np.array([0.0, 0.0], dtype=np.float32)

        with patch("airsim.environments.airsim_env.logger.warning") as warning_mock:
            env.step(action)

        assert not any(
            call.args and "Pause watchdog expired" in str(call.args[0])
            for call in warning_mock.call_args_list
        )

    def test_headless_watchdog_falls_back_to_continuous_stepping(self):
        """Repeated headless pause watchdog expiries should disable pause stepping."""
        mock_module, _ = _make_mock_airsim_module()
        env = _make_env(
            mock_module,
            training_mode=True,
            no_rendering_mode=True,
            use_pause_stepping=True,
        )
        env._physics_update_timeout = 0.0
        env.bridge.is_paused = MagicMock(return_value=False)

        env.step(np.array([0.0, 0.0], dtype=np.float32))
        assert env._force_continuous_stepping is False

        env.step(np.array([0.0, 0.0], dtype=np.float32))
        assert env._force_continuous_stepping is True

        env._wait_for_non_pause_state_update = MagicMock(
            return_value=(env.bridge.get_car_state(), 0.01)
        )
        env.step(np.array([0.0, 0.0], dtype=np.float32))
        assert env._wait_for_non_pause_state_update.called
