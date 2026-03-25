"""Tests for AirSimBridge simulation-speed bookkeeping."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

_SRC_DIR = Path(__file__).resolve().parents[3] / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from airsim.communication.airsim_bridge import AirSimBridge


def test_prepare_fossen_visualisation_updates_internal_simulation_speed(monkeypatch):
    """Preparing visualisation should update the cached bridge speed value."""
    bridge = AirSimBridge.__new__(AirSimBridge)
    bridge._connected = True
    bridge._last_health_check = 0.0
    bridge.config = SimpleNamespace(health_check_interval=10)
    bridge._simulation_speed = 1.0
    bridge._controls = None
    bridge.coord_transform = None
    bridge.render_z = 0.0
    client_speed_values = []
    bridge._client = SimpleNamespace(
        simGetVehiclePose=lambda: SimpleNamespace(position=SimpleNamespace(z_val=12.5)),
        setCarControls=lambda controls: None,
        simSetSimulationSpeed=lambda value: client_speed_values.append(float(value)),
    )
    bridge._client_speed = None

    bridge._airsim = SimpleNamespace(CarControls=lambda: SimpleNamespace(throttle=0.0, steering=0.0, brake=0.0))

    monkeypatch.setattr(bridge, "load_world_save", lambda save_slot: None)
    monkeypatch.setattr(AirSimBridge, "_ensure_connected", lambda self: True)

    def _fake_load_saved_calibration_reference(client, draw_reference=True):
        """Return a dummy coordinate transform and metadata payload.

        Args:
            client: Fake AirSim client instance.
            draw_reference: Whether overlay drawing is enabled.

        Returns:
            A tuple matching the calibration-loader contract.
        """
        return object(), {"draw_reference": draw_reference}

    import sys as _sys

    module_name = "fw.common.airsim.tools.calibrate_coords"
    previous_module = _sys.modules.get(module_name)
    _sys.modules[module_name] = SimpleNamespace(
        load_saved_calibration_reference=_fake_load_saved_calibration_reference
    )
    try:
        bridge.prepare_fossen_visualisation(simulation_speed=20.0)
    finally:
        if previous_module is None:
            del _sys.modules[module_name]
        else:
            _sys.modules[module_name] = previous_module

    assert bridge._simulation_speed == 20.0
    assert client_speed_values == [20.0]
