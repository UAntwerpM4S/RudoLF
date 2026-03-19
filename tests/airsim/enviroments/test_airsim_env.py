"""Unit tests for AirSimEnv (T-009 to T-012).

All tests are fully mocked — no UE5 or CosysAirSim is required.
"""

import csv
import importlib
import sys
import pytest
import numpy as np
from unittest.mock import MagicMock, patch


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


def _make_env(mock_module):
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

        env = AirSimEnv(
            ship_pos=[2245, -47],
            target_pos=[-1559, 11990],
            max_steps=100,
            use_pause_stepping=False,
            training_mode=False,
            verbose=False,
        )
    return env


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


# ── T-009: Observation mapping & normalisation ─────────────


class TestObservationSpace:
    """T-009: Verify observation space shape, bounds, and normalisation."""

    def test_observation_space_shape(self, mock_airsim):
        """Observation space must be Box(10,) matching PySimEnv."""
        env, _ = mock_airsim
        assert env.observation_space.shape == (10,)
        assert env.observation_space.dtype == np.float32

    def test_observation_bounds(self, mock_airsim):
        """Low/high arrays must match the documented channel ranges."""
        env, _ = mock_airsim
        expected_low = np.array(
            [-1, -1, -1, 0, 0, -2, -1, -1, -1, -1], dtype=np.float32
        )
        expected_high = np.array(
            [1, 1, 1, 1, 1, 2, 1, 1, 1, 1], dtype=np.float32
        )
        np.testing.assert_array_equal(env.observation_space.low, expected_low)
        np.testing.assert_array_equal(env.observation_space.high, expected_high)

    def test_observation_within_bounds(self, mock_airsim):
        """Returned observation must always be clipped to obs-space."""
        env, _ = mock_airsim
        obs = env._get_obs()
        assert obs.shape == (10,)
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


# ── T-010: 2D (PySimEnv) vs 3D (AirSimEnv) observation parity ────────────


class TestObsSpaceParity:
    """T-010: Validate the current observation-space compatibility contract.

    AirSimEnv uses a flat 10-channel observation. PySimEnv currently exposes
    12 channels per ship (the AirSim-compatible core plus 2 extra features).
    """

    @pytest.fixture(scope="class")
    def pysim_env(self):
        """Create a real PySimEnv (pure Python, no mocking required)."""
        from fw.environments.py_sim_env import PySimEnv
        env = PySimEnv(render_mode=None, num_ships=1)
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
        """PySimEnv must expose at least the AirSim 10-channel core."""
        airsim_env, _ = mock_airsim
        py_obs_shape = pysim_env.observation_space.shape
        air_obs_shape = airsim_env.observation_space.shape

        n_channels_pysim = py_obs_shape[-1]
        n_channels_airsim = air_obs_shape[0]

        assert n_channels_airsim == 10
        assert n_channels_pysim == 12
        assert n_channels_pysim >= n_channels_airsim

    def test_obs_bounds_per_channel_match(self, mock_airsim, pysim_env):
        """Validate matching bounds for shared channels and known differences."""
        airsim_env, _ = mock_airsim
        py_low = pysim_env.observation_space.low.squeeze(axis=0)
        py_high = pysim_env.observation_space.high.squeeze(axis=0)

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
        """PySimEnv wraps observations per ship; AirSimEnv is flat.

        Callers that transfer policies between environments must handle this
        shape/channel difference explicitly.
        """
        airsim_env, _ = mock_airsim
        assert airsim_env.observation_space.shape == (10,)
        assert pysim_env.observation_space.shape == (1, 12)




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
