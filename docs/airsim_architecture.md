# AirSim Architecture

This document describes the current AirSim subsystem in this repository, including
the modular export pipeline used by inference and framework-authoritative UE5
visualisation runs.

## Current Module Layout

```text
src/airsim/
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
|   `-- visualisation_models/run_visualise_fossen.py (legacy compatibility)
|-- train/train_airsim.py
|-- poc/
`-- verantwoording_workflow.md

src/fw/common/airsim/
|-- classes/path_resolution.py
`-- tools/
    |-- angles.py
    |-- path_utils.py
    |-- calibrate_coords.py
    `-- analyze_telemetry.py

src/fw/common/tools/
`-- airsim_export.py

src/common_tools/
|-- telemetry_export.py
`-- telemetry_exporter.py
```

## Responsibilities

- `airsim_bridge.py`: UE5 RPC transport, reconnect, save-slot prep, controls forwarding, telemetry/state reads, and mirroring helpers.
- `control_mapping.py`: converts validated policy actions and actuator values into UE5-compatible controls.
- `coord_transform.py`: frame and heading transforms between framework coordinates and AirSim coordinates.
- `airsim_env.py`: Gymnasium environment that runs against UE5/CosysAirSim.
- `run_inference.py`: direct UE5 inference using `AirSimEnv`.
- `run_visualise_models.py`: framework replay mirrored into UE5 with optional
  minimap rendering and paced replay controls.
- `fw/common/airsim/tools/*`: shared AirSim utilities used by framework and AirSim entrypoints.
- `fw/common/tools/airsim_export.py`: session export helpers for inference and
  visualisation rows.
- `common_tools/telemetry_exporter.py`: buffered CSV/metadata writer used by
  session exporters.

## Runtime Flow

### Entrypoint control

- `fw.main` is the ordered framework entrypoint.
- For each configured `AirSimEnv` step, `fw.main` delegates to `fw.main_with_airsim`.
- Non-AirSim steps stay in generic framework training/visualisation flow.

### AirSim-specific dispatch

`fw.main_with_airsim` reads `airsim/config/runtime_settings.json`:

- `visualization.fossen_visualise = true` -> `airsim.inference.run_visualise_models`.
- `visualization.inference_visualise = true` -> `airsim.inference.run_inference`.
- if both are true, Fossen visualisation is prioritized.

### Training path (UE5 authoritative)

1. `AirSimEnv` connects through `AirSimBridge`.
2. Policy action is validated/mapped via control mapping.
3. Controls are sent to UE5.
4. UE5 state is read back.
5. Observation/reward/termination are computed in `AirSimEnv`.
6. Session telemetry and metadata are exported under `exports/`.

### Replay path (framework authoritative)

1. Framework env (`PySimEnv`/`FhSimEnv`) is stepped.
2. State is transformed via `CoordTransform`.
3. Pose/kinematics/object updates are mirrored into UE5 through `AirSimBridge`.
4. Optional minimap rendering runs separately.
5. Session telemetry, bridge telemetry, and metadata are exported under `exports/`.

## Export Artifacts

Each AirSim run creates a timestamped session folder under the workspace-level
`exports/` directory.

Inference sessions contain:

- `inference_session.csv`
- `airsim_telemetry.csv`
- `metadata.json`

Visualisation sessions contain:

- `visualisation_session.csv`
- `airsim_telemetry.csv`
- `metadata.json`

`metadata.json` records the used model, selected environment, and runtime
configuration snapshot for the session.

## Key Configuration Layers

- Framework selection: `src/fw/config.py` (`env_type`, `action`).
- AirSim runtime behavior: `src/airsim/config/runtime_settings.json` (`airsim`, `training`, `visualization`).
- Connection/config schema: `src/airsim/config/airsim_config.py`.

Important visualisation runtime controls:

- `visualization.save_matplotlib_minimap`
- `visualization.min_log_interval_s`
- `visualization.minimap_update_hz`
- `visualization.bridge_telemetry_hz`
- `visualization.startup_settle_s`

## Testing Map

AirSim-specific tests:

- `tests/airsim/communication/test_bridge.py`
- `tests/airsim/communication/test_control_mapping.py`
- `tests/airsim/communication/test_airsim_bridge.py`
- `tests/airsim/enviroments/test_airsim_env.py`
- `tests/airsim/inference/test_fossen_visualisation_dispatch.py`
- `tests/airsim/inference/test_fossen_visualisation_timing.py`
- `tests/airsim/inference/test_minimap_layout.py`
- `tests/airsim/inference/test_visualise_models_minimap_save.py`
- `tests/fw/common/tools/test_airsim_export.py`
- `tests/test_telemetry_exporter.py`

Framework/AirSim routing tests:

- `tests/fw/test_main_fallback.py`
- `tests/fw/test_env_factory_config.py`

## Related Documentation

- `src/airsim/README.md`
- `src/airsim/inference/README.md`
- `src/airsim/verantwoording_workflow.md`
