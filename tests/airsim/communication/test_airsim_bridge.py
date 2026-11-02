"""
Tests for AirSimBridge simulation speed, rendering toggle and output silencing.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

_SRC_DIR = Path(__file__).resolve().parents[3] / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from airsim.communication.airsim_bridge import AirSimBridge, _silence_library_output


def test_prepare_model_visualisation_updates_internal_simulation_speed(monkeypatch):
    """
    prepare_model_visualisation sets the client and cached simulation speed and enables rendering.
    """

    bridge = AirSimBridge.__new__(AirSimBridge)
    bridge._connected = True
    bridge._last_health_check = 0.0
    bridge.config = SimpleNamespace(health_check_interval=10)
    bridge._simulation_speed = 1.0
    bridge._rendering_enabled = None
    bridge._controls = None
    bridge.coord_transform = None
    bridge.render_z = 0.0
    client_speed_values = []
    console_commands = []
    bridge._client = SimpleNamespace(
        simGetVehiclePose=lambda: SimpleNamespace(position=SimpleNamespace(z_val=12.5)),
        setCarControls=lambda controls: None,
        simSetSimulationSpeed=lambda value: client_speed_values.append(float(value)),
        simRunConsoleCommand=lambda command: console_commands.append(command) or True,
    )
    bridge._client_speed = None

    bridge._airsim = SimpleNamespace(CarControls=lambda: SimpleNamespace(throttle=0.0, steering=0.0, brake=0.0))

    monkeypatch.setattr(bridge, "load_world_save", lambda save_slot: None)
    monkeypatch.setattr(AirSimBridge, "_ensure_connected", lambda self: True)

    def _fake_load_saved_calibration_reference(client, draw_reference=True, duration=10.0):
        """
        Fake calibration loader: (transform, metadata).
        """

        return object(), {"draw_reference": draw_reference}

    import sys as _sys

    module_name = "fw.common.airsim.tools.calibrate_coords"
    previous_module = _sys.modules.get(module_name)
    _sys.modules[module_name] = SimpleNamespace(
        load_saved_calibration_reference=_fake_load_saved_calibration_reference
    )
    try:
        bridge.prepare_model_visualisation(simulation_speed=20.0)
    finally:
        if previous_module is None:
            del _sys.modules[module_name]
        else:
            _sys.modules[module_name] = previous_module

    assert bridge._simulation_speed == 20.0
    assert client_speed_values == [20.0]
    assert console_commands == ["ShowFlag.Rendering 1"]


def test_set_rendering_enabled_uses_console_command(monkeypatch):
    """
    Rendering is toggled via the UE5 console command.
    """

    bridge = AirSimBridge.__new__(AirSimBridge)
    bridge._connected = True
    bridge._last_health_check = 0.0
    bridge.config = SimpleNamespace(health_check_interval=10)
    bridge._rendering_enabled = None
    issued_commands = []
    bridge._client = SimpleNamespace(
        simRunConsoleCommand=lambda command: issued_commands.append(command) or True,
    )

    monkeypatch.setattr(AirSimBridge, "_ensure_connected", lambda self: True)

    assert bridge.set_rendering_enabled(False) is True
    assert bridge._rendering_enabled is False
    assert issued_commands == ["ShowFlag.Rendering 0"]


def test_silence_library_output_suppresses_stdout_and_stderr(capsys):
    """
    Library output on stdout and stderr is suppressed.
    """

    def _noisy_callback():
        """
        Write to stdout and stderr.
        """

        print("Client Ver:4")
        print("Server Ver:3", file=sys.stderr)
        return "ok"

    result = _silence_library_output(_noisy_callback)
    captured = capsys.readouterr()

    assert result == "ok"
    assert captured.out == ""
    assert captured.err == ""
