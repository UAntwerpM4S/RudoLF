"""Tests for the AirSimEnv-only inference runner."""

from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import torch


def _install_fake_matplotlib(monkeypatch) -> None:
    """Install a tiny fake matplotlib module for inference tests."""

    class _FakeAxes:
        """Minimal matplotlib axes stub used by the inference tests."""

        def set_title(self, *args, **kwargs):
            """No-op title setter."""

        def set_xlabel(self, *args, **kwargs):
            """No-op x-label setter."""

        def set_ylabel(self, *args, **kwargs):
            """No-op y-label setter."""

        def set_aspect(self, *args, **kwargs):
            """No-op aspect setter."""

        def plot(self, *args, **kwargs):
            """No-op line plot."""

        def legend(self, *args, **kwargs):
            """No-op legend."""

    matplotlib_module = types.ModuleType("matplotlib")
    pyplot_module = types.ModuleType("matplotlib.pyplot")
    pyplot_module.subplots = lambda *args, **kwargs: (object(), _FakeAxes())
    pyplot_module.tight_layout = lambda: None
    pyplot_module.savefig = lambda *args, **kwargs: None
    pyplot_module.show = lambda: None
    pyplot_module.cm = types.SimpleNamespace(
        tab10=types.SimpleNamespace(colors=["red", "blue", "green"]),
    )
    matplotlib_module.pyplot = pyplot_module

    monkeypatch.setitem(sys.modules, "matplotlib", matplotlib_module)
    monkeypatch.setitem(sys.modules, "matplotlib.pyplot", pyplot_module)


def test_run_inference_uses_native_airsim_episode_loop(monkeypatch):
    """Verify inference uses AirSimEnv.reset/step/close for all episodes."""
    import airsim.inference.run_inference as run_inference_module

    _install_fake_matplotlib(monkeypatch)

    resolved_model_path = Path("src/models/AirSimEnv_PPO_policy.zip")
    monkeypatch.setattr(
        run_inference_module,
        "resolve_model_path",
        lambda model_path, src_dir: resolved_model_path,
    )

    class _FakePolicy:
        """Minimal policy stub returning a deterministic action."""

        device = "cpu"

        def _predict(self, state_t, deterministic=True):
            """Return a constant action tensor."""
            return torch.tensor([[0.25, -0.5]], dtype=torch.float32), None, None

    monkeypatch.setattr(
        run_inference_module,
        "load_policy",
        lambda model_path, device="cpu": _FakePolicy(),
    )

    class _FakeExporter:
        """Capture logged rows and metadata for assertions."""

        instance = None

        def __init__(self, *args, **kwargs):
            """Store export state for the active test run."""
            self.rows = []
            self.metadata_updates = []
            self.saved = False
            self.closed = False
            self.export_dir = Path("exports/fake_session")
            self.bridge_log_path = self.export_dir / "airsim_telemetry.csv"
            _FakeExporter.instance = self

        def log_inference_step(self, row):
            """Store one logged inference row."""
            self.rows.append(row)

        def update_metadata(self, metadata=None, **kwargs):
            """Store metadata updates."""
            merged = dict(metadata or {})
            merged.update(kwargs)
            self.metadata_updates.append(merged)

        def save(self):
            """Record save invocation."""
            self.saved = True

        def close(self):
            """Record close invocation."""
            self.closed = True

        def __enter__(self):
            """Support context-manager usage like the real exporter."""
            return self

        def __exit__(self, exc_type, exc_value, traceback) -> None:
            """Mirror exporter cleanup on context-manager exit."""
            self.save()
            self.close()

    monkeypatch.setattr(run_inference_module, "AirSimSessionExporter", _FakeExporter)

    class _FakeEnv:
        """Minimal AirSimEnv double for episode-loop verification."""

        instance = None

        def __init__(self, **kwargs):
            """Capture constructor args and initialize fake env state."""
            self.constructor_kwargs = kwargs
            self.bridge = types.SimpleNamespace(is_connected=True)
            self.initial_ship_pos = np.array([2245.0, -47.0], dtype=np.float32)
            self.ship_pos = self.initial_ship_pos.copy()
            self.ship_heading = 0.0
            self.ship_speed = 0.0
            self.checkpoint_index = 1
            self.cross_error = 0.0
            self.last_physics_wait_s = 0.02
            self.last_step_latency_s = 0.03
            self._last_ue5_timestamp = 100
            self.last_controls = {"brake": 0.0}
            self.time_step = 0.1
            self.reset_calls = 0
            self.step_calls = 0
            self.close_calls = 0
            self._episode_step = 0
            _FakeEnv.instance = self

        def reset(self):
            """Start a new fake episode at the initial ship position."""
            self.reset_calls += 1
            self._episode_step = 0
            self.ship_pos = self.initial_ship_pos.copy()
            self.ship_heading = 0.0
            self.ship_speed = 0.0
            self.checkpoint_index = 1
            self.cross_error = 0.0
            return np.array([0.0, 1.0], dtype=np.float32), {}

        def step(self, action):
            """Advance one fake step and end episodes deterministically."""
            self.step_calls += 1
            self._episode_step += 1
            self.ship_pos = self.ship_pos + np.array([10.0, 5.0], dtype=np.float32)
            self.ship_heading += 0.05
            self.ship_speed = 1.2 + 0.1 * self._episode_step
            self.checkpoint_index += 1
            self.cross_error = 0.5 * self._episode_step
            self._last_ue5_timestamp += 1
            self.last_controls = {
                "steering": float(action[0]),
                "throttle": float(action[1]),
                "brake": 0.0,
            }

            terminated = self.reset_calls == 1 and self._episode_step == 2
            truncated = self.reset_calls == 2 and self._episode_step == 1
            reward = 3.0 if terminated else (-250.0 if truncated else 1.5)
            info = {
                "checkpoint_index": self.checkpoint_index,
                "cross_track_error": self.cross_error,
            }
            obs = np.array([self._episode_step, self.reset_calls], dtype=np.float32)
            return obs, reward, terminated, truncated, info

        def close(self):
            """Record shutdown."""
            self.close_calls += 1

    monkeypatch.setattr(run_inference_module, "AirSimEnv", _FakeEnv)
    monkeypatch.setattr(
        run_inference_module,
        "build_airsim_env_snapshot",
        lambda env: {
            "ship_x": float(env.ship_pos[0]),
            "ship_y": float(env.ship_pos[1]),
            "heading_deg": 12.0,
            "speed": float(env.ship_speed),
            "checkpoint_index": int(env.checkpoint_index),
            "cross_track_error": float(env.cross_error),
            "physics_wait_s": float(env.last_physics_wait_s),
            "step_latency_s": float(env.last_step_latency_s),
            "ue5_timestamp": int(env._last_ue5_timestamp),
        },
    )

    run_inference_module.run_inference(
        model_path=str(resolved_model_path),
        max_steps=5,
        num_episodes=2,
        device="cpu",
    )

    env = _FakeEnv.instance
    exporter = _FakeExporter.instance

    assert env is not None
    assert env.reset_calls == 2
    assert env.step_calls == 3
    assert env.close_calls == 1
    assert env.constructor_kwargs["training_mode"] is False
    assert env.constructor_kwargs["use_pause_stepping"] is False
    assert env.constructor_kwargs["simulation_speed"] == 1.0

    assert exporter is not None
    assert exporter.saved is True
    assert exporter.closed is True
    assert [row["outcome"] for row in exporter.rows] == [
        "running",
        "terminated",
        "truncated",
    ]
    assert all(row["control_mode"] == "airsim_env_step" for row in exporter.rows)
    assert any(
        update.get("last_outcome") == "truncated"
        for update in exporter.metadata_updates
    )


def test_build_parser_removes_legacy_mode_flags():
    """Verify the AirSim-only CLI no longer exposes legacy mode switches."""
    import airsim.inference.run_inference as run_inference_module

    parser = run_inference_module._build_parser()

    assert "--ue5" not in parser._option_string_actions
    assert "--fossen" not in parser._option_string_actions
    assert "--model" in parser._option_string_actions
