"""Tests for minimap save behavior in run_visualise_models."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import airsim.inference.run_visualise_models as module


class _FakeBridge:
    """Minimal bridge stub for visualisation tests."""

    def __init__(self, *args, **kwargs) -> None:
        """Initialise the fake bridge.

        Args:
            *args: Positional constructor arguments.
            **kwargs: Keyword constructor arguments.
        """
        self.config = args[0] if args else None
        self.is_connected = True
        self._controls = None
        self._airsim = SimpleNamespace(CarControls=lambda: SimpleNamespace(throttle=0.0, steering=0.0, brake=0.0))

    def prepare_fossen_visualisation(self, **kwargs) -> None:
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

    finished = False

    def __init__(self, env, save_dir=".", **kwargs) -> None:
        """Initialise the fake minimap.

        Args:
            env: Fake environment instance.
            save_dir: Ignored save directory.
            **kwargs: Ignored minimap options.
        """
        self.env = env

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
        type(self).finished = True

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
        self.telemetry_path = self.export_dir / (telemetry_filename or f"{session_type}_session.csv")
        self.bridge_log_path = self.export_dir / "airsim_telemetry.csv"
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


def test_visualise_models_does_not_save_matplotlib_minimap_by_default(monkeypatch):
    """Default visualisation should not save a matplotlib minimap snapshot."""
    monkeypatch.setattr(module, "load_policy", lambda *args, **kwargs: _FakePolicy())
    monkeypatch.setattr(module, "AirSimBridge", _FakeBridge)
    monkeypatch.setattr(module, "AirSimSessionExporter", _FakeExporter)
    monkeypatch.setattr(module, "resolve_model_path", lambda model_path, src_dir: Path(model_path))
    monkeypatch.setattr(module.VisualiseFossen, "_build_environment", lambda self, selected_env, max_steps: _FakeEnv())
    monkeypatch.setattr(module.VisualiseFossen, "_resolve_environment_name", lambda self, model_path: "PySimEnv")
    monkeypatch.setattr(module, "_framework_motion_state_from_env", lambda env, bridge: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    monkeypatch.setattr(module, "_animate_framework_state", lambda *args, **kwargs: ([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], None))
    monkeypatch.setattr(module, "VisualiseModelsMinimap", _FakeMinimap)

    _FakeMinimap.finished = False
    _FakeExporter.created = []
    module.run_visualise_models(
        model_path="models/PySimEnv_PPO_policy.zip",
        show_minimap=True,
        use_ue5_camera_minimap=False,
        num_episodes=1,
        max_steps=1,
    )

    assert _FakeMinimap.finished is False


def test_visualise_models_exports_rows_and_bridge_log_path(monkeypatch):
    """Visualisation should initialize an export session and emit telemetry rows."""
    monkeypatch.setattr(module, "load_policy", lambda *args, **kwargs: _FakePolicy())
    monkeypatch.setattr(module, "AirSimBridge", _FakeBridge)
    monkeypatch.setattr(module, "AirSimSessionExporter", _FakeExporter)
    monkeypatch.setattr(module, "resolve_model_path", lambda model_path, src_dir: Path(model_path))
    monkeypatch.setattr(module.VisualiseFossen, "_build_environment", lambda self, selected_env, max_steps: _FakeEnv())
    monkeypatch.setattr(module.VisualiseFossen, "_resolve_environment_name", lambda self, model_path: "PySimEnv")
    monkeypatch.setattr(module, "_framework_motion_state_from_env", lambda env, bridge: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    monkeypatch.setattr(module, "_animate_framework_state", lambda *args, **kwargs: ([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], None))

    _FakeExporter.created = []
    module.run_visualise_models(
        model_path="models/PySimEnv_PPO_policy.zip",
        show_minimap=False,
        num_episodes=1,
        max_steps=1,
    )

    exporter = _FakeExporter.created[-1]
    assert exporter.rows
    assert exporter.saved is True
    assert exporter.closed is True
    assert exporter.session_type == "visualisation"
    assert exporter.telemetry_path.name == "visualisation_session.csv"
    assert exporter.bridge_log_path.name == "airsim_telemetry.csv"
