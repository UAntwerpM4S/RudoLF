"""Tests for minimap save behavior in run_visualise_models."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np

import airsim.inference.run_visualise_models as module
from fw.config import (
    AIRSIM_BRIDGE_LOG_FILENAME,
    AIRSIM_SESSION_TELEMETRY_FILENAME_TEMPLATE,
)
from fw.policies.ppo_policy import PPOPolicy


class _FakeBridge:
    """Minimal bridge stub for visualisation tests."""

    created = []

    def __init__(self, *args, **kwargs) -> None:
        """Initialise the fake bridge.

        Args:
            *args: Positional constructor arguments.
            **kwargs: Keyword constructor arguments.
        """
        self.config = args[0] if args else None
        self.is_connected = True
        self._controls = None
        self._airsim = SimpleNamespace(
            CarControls=lambda: SimpleNamespace(throttle=0.0, steering=0.0, brake=0.0)
        )
        self.airsim = self._airsim
        type(self).created.append(self)

    @property
    def current_controls(self):
        """Expose the latest control payload via the public bridge API."""
        return self._controls

    def prepare_model_visualisation(self, **kwargs) -> None:
        """Pretend to prepare UE5 visualisation state.

        Args:
            **kwargs: Ignored visualisation options.
        """
        return None

    def set_controls(self, controls) -> None:
        """Capture the latest control object.

        Args:
            controls: Fake AirSim controls payload.
        """
        self._controls = controls

    def set_simulation_speed(self, value: float) -> None:
        """Accept a simulation speed update.

        Args:
            value: Requested simulation-speed multiplier.
        """
        return None

    def get_state(self):
        """Return a minimal fake telemetry payload."""
        return {"speed": 0.0}

    def disconnect(self) -> None:
        """Pretend to disconnect from AirSim."""
        return None


class _SmokeBridge(_FakeBridge):
    """Richer fake bridge that supports the real adapter animation path."""

    def __init__(self, *args, **kwargs) -> None:
        """Initialize the richer fake bridge used by the smoke test.

        Args:
            *args: Positional arguments forwarded to the base fake bridge.
            **kwargs: Keyword arguments forwarded to the base fake bridge.
        """
        super().__init__(*args, **kwargs)
        self.coord_transform = SimpleNamespace(
            fossen_to_airsim=lambda x, y, z: (x + 10.0, y - 5.0, z),
            fossen_heading_to_airsim_yaw=lambda heading: heading + 0.1,
            fossen_velocity_to_airsim=lambda vx, vy: (vx, vy),
            fossen_yaw_rate_to_airsim=lambda yaw_rate: yaw_rate,
        )
        self.render_z = -3.0
        self.pose_calls = []
        self.trail_calls = []
        self.airsim = SimpleNamespace(
            CarControls=lambda: SimpleNamespace(throttle=0.0, steering=0.0, brake=0.0),
            Vector3r=lambda x, y, z: SimpleNamespace(x_val=x, y_val=y, z_val=z),
        )

    def set_pose(self, x: float, y: float, z: float, yaw: float, **kwargs) -> None:
        """Capture pose updates emitted by the visualiser."""
        self.pose_calls.append((x, y, z, yaw, kwargs))

    def plot_line_strip(self, points, color_rgba, thickness, is_persistent=True) -> None:
        """Capture trail plot calls."""
        self.trail_calls.append((points, color_rgba, thickness, is_persistent))


class _FakeEnv:
    """Minimal environment stub for visualisation tests."""

    def __init__(self) -> None:
        """Initialise a one-step fake framework environment."""
        self.ship_pos = [[0.0, 0.0]]
        self.phys_sims = [SimpleNamespace(state=SimpleNamespace(u=0.0, heading=0.0))]
        self.time_step = 0.1
        self.fh_sim = SimpleNamespace(dispose=lambda: None)
        self._done_once = False

    def reset(self):
        """Reset the fake environment state.

        Returns:
            A minimal observation/info pair.
        """
        self._done_once = False
        return [0.0], {}

    def step(self, action):
        """Advance the fake environment by one step.

        Args:
            action: Ignored policy action payload.

        Returns:
            A Gym-style step tuple.
        """
        if self._done_once:
            return [0.0], 0.0, True, False, {}
        self._done_once = True
        return [0.0], 1.0, True, False, {}


class _SmokeEnv:
    """Lightweight framework env that exercises the real adapter path."""

    def __init__(self) -> None:
        """Initialize the lightweight environment used by the smoke test."""
        self.ship_pos = np.array([100.0, 200.0], dtype=np.float32)
        self.phys_sims = [SimpleNamespace(state=SimpleNamespace(heading=0.2, u=1.5, v=0.1, r=0.05))]
        self.performed_action = np.array([0.2, 0.6], dtype=np.float32)
        self.ships = [SimpleNamespace(specifications=SimpleNamespace(max_rudder_angle=1.0, min_thrust=-1.0, max_thrust=1.0))]
        self.time_step = 0.1
        self.fh_sim = SimpleNamespace(dispose=lambda: None)
        self._done_once = False

    def reset(self):
        """Reset the fake smoke environment to its initial state.

        Returns:
            tuple[np.ndarray, dict]: Initial observation and empty info dict.
        """
        self._done_once = False
        self.ship_pos = np.array([100.0, 200.0], dtype=np.float32)
        self.phys_sims[0].state = SimpleNamespace(heading=0.2, u=1.5, v=0.1, r=0.05)
        self.performed_action = np.array([0.2, 0.6], dtype=np.float32)
        return np.array([0.0], dtype=np.float32), {}

    def step(self, action):
        """Advance the fake smoke environment by one deterministic step.

        Args:
            action: Action vector chosen by the loaded policy.

        Returns:
            tuple[np.ndarray, float, bool, bool, dict]: Gym-style step tuple.
        """
        self._done_once = True
        self.ship_pos = np.array([101.0, 202.0], dtype=np.float32)
        self.phys_sims[0].state = SimpleNamespace(heading=0.25, u=1.8, v=0.05, r=0.04)
        self.performed_action = np.array([float(action[0]), float(action[1])], dtype=np.float32)
        return np.array([0.1], dtype=np.float32), 1.25, True, False, {}


class _BatchedObsSmokeEnv(_SmokeEnv):
    """Smoke env variant returning a ship-batched observation layout."""

    def reset(self):
        """Return a batched reset observation shaped like ``(1, obs_dim)``."""
        obs, info = super().reset()
        return obs.reshape(1, -1), info

    def step(self, action):
        """Return a batched step observation shaped like ``(1, obs_dim)``."""
        obs, reward, done, truncated, info = super().step(action)
        return obs.reshape(1, -1), reward, done, truncated, info


class _FakePolicy:
    """Policy stub returning a deterministic action."""

    def predict(self, obs, deterministic=True):
        """Return a deterministic zero action.

        Args:
            obs: Ignored observation payload.
            deterministic: Ignored deterministic flag.

        Returns:
            A fake prediction tuple matching the policy API.
        """
        return [0.0, 0.0], None, None


class _FakeMinimap:
    """Minimap stub exposing finish/close for verification."""

    created = []

    def __init__(self, env, save_dir=".", **kwargs) -> None:
        """Initialise the fake minimap.

        Args:
            env: Fake environment instance.
            save_dir: Ignored save directory.
            **kwargs: Ignored minimap options.
        """
        self.env = env
        self.finished = False
        type(self).created.append(self)

    def update(self, env) -> None:
        """Accept a minimap update call.

        Args:
            env: Fake environment instance.
        """
        return None

    def finish(self, total_reward: float, episode: int = 1) -> None:
        """Record that a save/finish action was requested.

        Args:
            total_reward: Episode reward.
            episode: One-based episode index.
        """
        self.finished = True

    def close(self) -> None:
        """Close the fake minimap."""
        return None


class _FakeExporter:
    """Exporter stub capturing visualisation rows without touching disk."""

    created = []

    def __init__(
        self,
        session_name,
        metadata=None,
        export_dir=None,
        flush_interval=25,
        session_type="inference",
        telemetry_filename=None,
    ) -> None:
        """Initialise the fake exporter.

        Args:
            session_name: Requested session label.
            metadata: Optional metadata payload.
            export_dir: Optional export directory.
            flush_interval: Ignored flush interval.
            session_type: Session type used for the telemetry filename.
            telemetry_filename: Optional explicit telemetry filename.
        """
        self.session_name = session_name
        self.session_type = session_type
        self.metadata = dict(metadata or {})
        self.export_dir = Path(export_dir or "exports") / session_name
        self.telemetry_path = self.export_dir / (
            telemetry_filename
            or AIRSIM_SESSION_TELEMETRY_FILENAME_TEMPLATE.format(session_type=session_type)
        )
        self.bridge_log_path = self.export_dir / AIRSIM_BRIDGE_LOG_FILENAME
        self.rows = []
        self.saved = False
        self.closed = False
        type(self).created.append(self)

    def log_step(self, row) -> None:
        """Capture one telemetry row.

        Args:
            row: Telemetry payload to capture.
        """
        self.rows.append(dict(row))

    def log_inference_step(self, row) -> None:
        """Capture one inference telemetry row.

        Args:
            row: Telemetry payload to capture.
        """
        self.log_step(row)

    def log_visualisation_step(self, row) -> None:
        """Capture one visualisation telemetry row.

        Args:
            row: Telemetry payload to capture.
        """
        self.log_step(row)

    def update_metadata(self, metadata=None, **kwargs) -> None:
        """Merge metadata into the fake exporter state.

        Args:
            metadata: Optional metadata mapping.
            **kwargs: Additional metadata fields.
        """
        if metadata:
            self.metadata.update(metadata)
        self.metadata.update(kwargs)

    def save(self) -> None:
        """Mark the fake exporter as saved."""
        self.saved = True

    def close(self) -> None:
        """Mark the fake exporter as closed."""
        self.closed = True

    def __enter__(self):
        """Support exporter context-manager usage."""
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        """Mirror real exporter cleanup semantics."""
        self.save()
        self.close()


def test_visualise_models_does_not_save_matplotlib_minimap_by_default(monkeypatch):
    """Default visualisation should not save a matplotlib minimap snapshot."""
    monkeypatch.setattr(module.VisualiseModelsRunner, "_load_policy", lambda self, model_path, device: _FakePolicy())
    monkeypatch.setattr(module, "AirSimBridge", _FakeBridge)
    monkeypatch.setattr(module, "AirSimSessionExporter", _FakeExporter)
    monkeypatch.setattr(module, "resolve_model_path", lambda model_path, src_dir: Path(model_path))
    monkeypatch.setattr(module.VisualiseModelsRunner, "_build_environment", lambda self, selected_env, max_steps: _FakeEnv())
    monkeypatch.setattr(module.VisualiseModelsRunner, "_resolve_environment_name", lambda self, model_path: "PySimEnv")
    monkeypatch.setattr(module.VisualiseModelsRunner, "_motion_state_from_env", lambda self, env, bridge: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    monkeypatch.setattr(module.VisualiseModelsRunner, "_animate_state", lambda self, env, bridge, **kwargs: ([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], None))
    monkeypatch.setattr(module, "VisualiseModelsMinimap", _FakeMinimap)

    _FakeBridge.created = []
    _FakeMinimap.created = []
    _FakeExporter.created = []
    module.run_visualise_models(
        model_path="models/PySimEnv_PPO_policy.zip",
        show_minimap=True,
        use_ue5_camera_minimap=False,
        num_episodes=1,
        max_steps=1,
    )

    assert len(_FakeMinimap.created) == 1
    assert _FakeMinimap.created[0].finished is False
    assert _FakeBridge.created[-1]._controls is not None


def test_visualise_models_exports_rows_and_bridge_log_path(monkeypatch):
    """Visualisation should initialize an export session and emit telemetry rows."""
    monkeypatch.setattr(module.VisualiseModelsRunner, "_load_policy", lambda self, model_path, device: _FakePolicy())
    monkeypatch.setattr(module, "AirSimBridge", _FakeBridge)
    monkeypatch.setattr(module, "AirSimSessionExporter", _FakeExporter)
    monkeypatch.setattr(module, "resolve_model_path", lambda model_path, src_dir: Path(model_path))
    monkeypatch.setattr(module.VisualiseModelsRunner, "_build_environment", lambda self, selected_env, max_steps: _FakeEnv())
    monkeypatch.setattr(module.VisualiseModelsRunner, "_resolve_environment_name", lambda self, model_path: "PySimEnv")
    monkeypatch.setattr(module.VisualiseModelsRunner, "_motion_state_from_env", lambda self, env, bridge: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    monkeypatch.setattr(module.VisualiseModelsRunner, "_animate_state", lambda self, env, bridge, **kwargs: ([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], None))

    _FakeBridge.created = []
    _FakeExporter.created = []
    module.run_visualise_models(
        model_path="models/PySimEnv_PPO_policy.zip",
        show_minimap=False,
        num_episodes=1,
        max_steps=1,
    )

    exporter = _FakeExporter.created[-1]
    assert exporter.rows
    row = exporter.rows[0]
    assert isinstance(row, dict)
    assert "reward" in row
    assert "outcome" in row
    assert exporter.saved is True
    assert exporter.closed is True
    assert exporter.session_type == "visualisation"
    assert exporter.telemetry_path.name == AIRSIM_SESSION_TELEMETRY_FILENAME_TEMPLATE.format(session_type="visualisation")
    assert exporter.bridge_log_path.name == AIRSIM_BRIDGE_LOG_FILENAME
    assert _FakeBridge.created[-1]._controls is not None


def test_visualise_models_saves_matplotlib_minimap_when_enabled(monkeypatch):
    """Visualisation should finish the minimap when save is enabled."""
    monkeypatch.setattr(module.VisualiseModelsRunner, "_load_policy", lambda self, model_path, device: _FakePolicy())
    monkeypatch.setattr(module, "AirSimBridge", _FakeBridge)
    monkeypatch.setattr(module, "AirSimSessionExporter", _FakeExporter)
    monkeypatch.setattr(module, "resolve_model_path", lambda model_path, src_dir: Path(model_path))
    monkeypatch.setattr(module.VisualiseModelsRunner, "_build_environment", lambda self, selected_env, max_steps: _FakeEnv())
    monkeypatch.setattr(module.VisualiseModelsRunner, "_resolve_environment_name", lambda self, model_path: "PySimEnv")
    monkeypatch.setattr(module.VisualiseModelsRunner, "_motion_state_from_env", lambda self, env, bridge: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    monkeypatch.setattr(module.VisualiseModelsRunner, "_animate_state", lambda self, env, bridge, **kwargs: ([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], None))
    monkeypatch.setattr(module, "VisualiseModelsMinimap", _FakeMinimap)

    _FakeBridge.created = []
    _FakeMinimap.created = []
    _FakeExporter.created = []
    module.run_visualise_models(
        model_path="models/PySimEnv_PPO_policy.zip",
        show_minimap=True,
        use_ue5_camera_minimap=False,
        save_matplotlib_minimap=True,
        num_episodes=1,
        max_steps=1,
    )

    assert len(_FakeMinimap.created) == 1
    assert _FakeMinimap.created[0].finished is True


def test_visualise_models_smoke_path_uses_real_saved_policy(monkeypatch, tmp_path: Path):
    """Smoke-test the real adapter + PPOPolicy.load path with a fake bridge."""
    model_stem = tmp_path / "PySimEnv_PPO_policy"
    PPOPolicy(input_dim=1, output_dim=2, device="cpu").save(str(model_stem))

    monkeypatch.setattr(module, "AirSimBridge", _SmokeBridge)
    monkeypatch.setattr(module, "AirSimSessionExporter", _FakeExporter)
    monkeypatch.setattr(module, "resolve_model_path", lambda model_path, src_dir: model_stem.with_suffix(".zip"))
    monkeypatch.setattr(module.VisualiseModelsRunner, "_build_environment", lambda self, selected_env, max_steps: _SmokeEnv())

    _SmokeBridge.created = []
    _FakeExporter.created = []
    module.run_visualise_models(
        model_path=str(model_stem.with_suffix(".zip")),
        show_minimap=False,
        num_episodes=1,
        max_steps=1,
        save_matplotlib_minimap=False,
    )

    bridge = _SmokeBridge.created[-1]
    exporter = _FakeExporter.created[-1]

    assert bridge.pose_calls
    assert bridge.current_controls is not None
    assert exporter.rows
    assert exporter.rows[0]["model_origin"] == "pysim"


def test_visualise_models_smoke_path_supports_batched_single_ship_obs(monkeypatch, tmp_path: Path):
    """Visualisation should accept batched single-ship observations from envs."""
    model_stem = tmp_path / "PySimEnv_PPO_policy"
    PPOPolicy(input_dim=1, output_dim=2, device="cpu").save(str(model_stem))

    monkeypatch.setattr(module, "AirSimBridge", _SmokeBridge)
    monkeypatch.setattr(module, "AirSimSessionExporter", _FakeExporter)
    monkeypatch.setattr(module, "resolve_model_path", lambda model_path, src_dir: model_stem.with_suffix(".zip"))
    monkeypatch.setattr(module.VisualiseModelsRunner, "_build_environment", lambda self, selected_env, max_steps: _BatchedObsSmokeEnv())

    _SmokeBridge.created = []
    _FakeExporter.created = []
    module.run_visualise_models(
        model_path=str(model_stem.with_suffix(".zip")),
        show_minimap=False,
        num_episodes=1,
        max_steps=1,
        save_matplotlib_minimap=False,
    )

    bridge = _SmokeBridge.created[-1]
    exporter = _FakeExporter.created[-1]

    assert bridge.pose_calls
    assert bridge.current_controls is not None
    assert exporter.rows
