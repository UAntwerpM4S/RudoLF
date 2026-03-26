# AirSim Inference And Visualisation Runbook

This folder contains the two runtime entrypoints that stay intentionally separate:

- `run_inference.py`
- `run_visualise_models.py`

## Choose The Right Runner

### Use `run_inference.py` when

- the model was trained directly in `AirSimEnv`
- UE5 is the authoritative simulator
- you want the native training-style loop during inference
- you want episode resets, reward, checkpoint progress, and truncation to behave like `AirSimEnv`

### Use `run_visualise_models.py` when

- the model was trained in `PySimEnv` or `FhSimEnv`
- framework physics should remain authoritative
- UE5 is only used as renderer / mirror
- you want the replay minimap tooling

## Runtime Decision Path

`fw.main_with_airsim` reads `src/airsim/config/runtime_settings.json`.

Decision rules:

- `visualization.model_visualise=true` -> run `run_visualise_models.py`
- `visualization.fossen_visualise=true` -> same as above, legacy alias
- `visualization.inference_visualise=true` -> run `run_inference.py`
- if both replay and inference flags are true, replay wins

## `run_inference.py`

### Scope

`run_inference.py` is now AirSim-only.

It is meant for PPO policies trained inside the UE5-backed `AirSimEnv`. It no longer contains a Fossen compatibility path or a separate direct-control mode.

### Effective control loop

1. load the PPO model
2. construct `AirSimEnv` in inference mode
3. call `env.reset()` at the start of each episode
4. predict an action from the current observation
5. call `env.step(action)`
6. let `AirSimEnv.step()` handle:
   - action validation
   - action smoothing
   - control mapping
   - control send to UE5
   - physics wait
   - state readback
   - observation update
   - reward calculation
   - checkpoint progress
   - termination / truncation detection
7. stop the episode when `terminated` or `truncated` becomes true
8. call `env.close()` on shutdown

### What this means for tasks

- `T-026` is handled through the native `AirSimEnv.step()` action -> physics -> feedback loop
- `T-027` is handled through `AirSimEnv.reset()`, `AirSimEnv.step()`, and `AirSimEnv.close()`

## `run_visualise_models.py`

### Effective replay loop

1. resolve adapter from model family and environment
2. build framework env
3. load PPO policy
4. step framework env locally
5. convert motion to AirSim coordinates
6. mirror pose/kinematics/object state into UE5
7. update minimap if enabled
8. export telemetry bundle

### Supported adapters

Current adapter families:

- `fossen`
- `fhsim`

The adapter registry lives in `visualisation_models/model_adapters.py`.

## Minimap Options

### Matplotlib minimap

`VisualiseModelsMinimap` is a child-process matplotlib minimap based on `MinimapFossenAirsim`.

It currently shows:

- real-time ship marker
- heading line
- path trail
- ideal path
- waterway polygon
- Western Scheldt outline

### UE5 camera minimap

`UE5TopDownMinimap` uses a top-down AirSim camera and can optionally show an OpenCV preview window.

## Exports

Both runners create a timestamped session folder in `exports/`.

Typical files:

- `metadata.json`
- `airsim_telemetry.csv`
- `inference_session.csv` or `visualisation_session.csv`

## Text Overlay Status

UE5 text overlays are currently disabled.

That means:

- no HUD status banner
- no in-world status text near the ship

The runners now rely on console logging and exported telemetry instead.

## Commands

From `src/`:

### Live AirSimEnv inference

```bash
python -m airsim.inference.run_inference --model models/AirSimEnv_PPO_policy.zip
```

### Framework replay in UE5

```bash
python -m airsim.inference.run_visualise_models --model models/PySimEnv_PPO_policy.zip
```

## Important Runtime Settings

### Connection

- `airsim.ip`
- `airsim.port`
- `airsim.save_slot`
- `airsim.simulation_speed`

### Replay selection

- `visualization.model_visualise`
- `visualization.fossen_visualise`
- `visualization.model_family`
- `visualization.environment_name`

### Inference selection

- `visualization.inference_visualise`
- `visualization.inference_model_path`

### Replay rendering and minimap

- `visualization.show_minimap`
- `visualization.use_ue5_camera_minimap`
- `visualization.save_matplotlib_minimap`
- `visualization.minimap_update_hz`
- `visualization.bridge_telemetry_hz`
- `visualization.render_interpolation_steps`
- `visualization.use_kinematics_in_ue5`
- `visualization.vehicle_object_name`

## Troubleshooting

### UE5 unreachable

- ensure UE5 is running in Play mode
- verify `airsim.ip` and `airsim.port`

### Wrong runner starts

- check `visualization.model_visualise`
- check `visualization.inference_visualise`
- remember replay wins if both are enabled

### No minimap window appears

- set `visualization.show_minimap=true`
- for camera mode also set `visualization.use_ue5_camera_minimap=true`

### Framework-trained model passed to `run_inference.py`

- switch to `run_visualise_models.py`
- `run_inference.py` is reserved for models trained in `AirSimEnv`

### `plot_airsim_path.py`

This tool still exists as a standalone telemetry plotter. It is not wired into the main runtime paths automatically.
