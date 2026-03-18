# AirSim Integration Package

CosysAirSim bridge, Gymnasium environment, coordinate calibration, and inference/visualisation tooling for Unreal Engine 5 integration.

## Overview

The `src/airsim` package integrates the RudoLF framework with UE5/CosysAirSim.

It supports two runtime patterns:

- UE5-native training/inference with `AirSimEnv` as the physics source of truth.
- Framework-authoritative replay where `PySimEnv` or `FhSimEnv` is mirrored into UE5 for visualisation.

## Related Docs

- [AirSim Inference and Visualisation Runbook](inference/README.md)
- [Workflow Rationale (`fw.main` vs `fw.main_with_airsim`)](verantwoording_workflow.md)
- [Repository-level AirSim Architecture](../../docs/airsim_architecture.md)

## Package Structure

```text
airsim/
|-- __init__.py
|-- communication/
|   |-- airsim_bridge.py
|   |-- control_mapping.py
|   `-- coord_transform.py
|-- config/
|   |-- airsim_config.py
|   |-- calibration.json
|   |-- runtime_settings.json
|   `-- settings_training.json
|-- environments/
|   `-- airsim_env.py
|-- inference/
|   |-- README.md
|   |-- minimap_fossen_airsim.py
|   |-- minimap_models_ue5.py
|   |-- plot_airsim_path.py
|   |-- run_inference.py
|   |-- run_visualise_models.py
|   |-- verify_headless_plot.py
|   `-- visualisation_models/
|       `-- run_visualise_fossen.py  (legacy compatibility path)
|-- train/
|   `-- train_airsim.py
|-- poc/
|   |-- README.md
|   |-- airsim_bridge_PID.py
|   `-- train_airsim.py
`-- verantwoording_workflow.md
```

Shared AirSim utilities are now hosted under:

```text
src/fw/common/airsim/
|-- classes/path_resolution.py
`-- tools/{angles.py, path_utils.py, calibrate_coords.py, analyze_telemetry.py}
```

## Architecture Summary

### Core components

- `communication/airsim_bridge.py`: UE5 RPC lifecycle, reconnect behavior, controls forwarding, state reads, and mirroring methods.
- `communication/control_mapping.py`: policy-to-UE5 control mapping and validation helpers.
- `communication/coord_transform.py`: Fossen <-> AirSim coordinate and heading transforms.
- `environments/airsim_env.py`: Gymnasium environment backed by UE5/CosysAirSim.
- `inference/run_inference.py`: policy inference loop using `AirSimEnv`.
- `inference/run_visualise_models.py`: framework replay mirrored into UE5.

### Runtime entrypoints

- `python -m fw.main`
  - Ordered framework entrypoint.
  - Delegates each `AirSimEnv` step to `fw.main_with_airsim` at its list position.
- `python -m fw.main_with_airsim`
  - AirSim-specialized entrypoint.
  - Uses `runtime_settings.json` to select and configure AirSim behavior.

### Visualisation dispatch

In `fw.main_with_airsim`, runtime flags under `visualization` choose mode:

- `fossen_visualise = true` -> runs `airsim.inference.run_visualise_models`.
- `inference_visualise = true` -> runs `airsim.inference.run_inference`.
- If both are true, Fossen visualisation is prioritized.

## Configuration

### 1) Framework-level selection (`src/fw/config.py`)

- `env_type`: single value or ordered list.
- `action`: `train` or `visualize`.

### 2) AirSim runtime configuration (`src/airsim/config/runtime_settings.json`)

Key sections:

- `airsim`: bridge connectivity and simulation runtime values.
- `training`: speed profile and stepping controls.
- `visualization`: visualisation mode and replay options.

Important training controls:

- `training.speed_profile`: `normal`, `safe`, `fast`, `turbo`, or `custom`.
- `training.simulation_speed`
- `training.use_pause_stepping`
- `training.frames_per_step`
- `training.physics_update_timeout`

## Coordinate Calibration and Telemetry Utilities

Utilities are provided via `fw.common.airsim.tools`.

### Calibrate coordinate transform

```bash
cd src
python -m fw.common.airsim.tools.calibrate_coords
```

### Analyze bridge telemetry

```bash
cd src
python -m fw.common.airsim.tools.analyze_telemetry
```

## Common Workflows

### Train through ordered framework flow

```bash
cd src
python -m fw.main
```

### Train AirSim directly

```bash
cd src
python -m fw.main_with_airsim
```

### Run inference in UE5

```bash
cd src
python -m airsim.inference.run_inference --model models/AirSimEnv_PPO_policy.zip
```

### Replay framework model in UE5

```bash
cd src
python -m airsim.inference.run_visualise_models --model models/PySimEnv_PPO_policy.zip
```

Legacy wrapper path remains available for compatibility:

```bash
python -m airsim.inference.visualisation_models.run_visualise_fossen --model models/PySimEnv_PPO_policy.zip
```

## Tests

Run all AirSim tests:

```bash
python -m pytest tests/airsim -v
```

Important test modules:

- `tests/airsim/communication/test_bridge.py`
- `tests/airsim/communication/test_control_mapping.py`
- `tests/airsim/enviroments/test_airsim_env.py`
- `tests/airsim/inference/test_fossen_visualisation_dispatch.py`
- `tests/airsim/inference/test_fossen_visualisation_timing.py`
- `tests/airsim/inference/test_minimap_layout.py`

Framework routing tests that cover AirSim sequencing/fallback:

- `tests/fw/test_main_fallback.py`
- `tests/fw/test_env_factory_config.py`

## Notes

- `poc/` contains historical snapshots and is not the reference implementation for current runtime behavior.
- `runtime_settings.json` is strict JSON; explanatory notes are stored under `_remarks` fields.
