# AirSim Inference And Visualisation Runbook

This folder contains the two runtime entrypoints that stay intentionally separate:

- `run_inference.py`
- `run_visualise_models.py`

Related docs:

- [AirSim Step-By-Step Setup Guide](../../../docs/airsim_setup_step_by_step.md)
- [AirSim Architecture](../../../docs/airsim_architecture.md)

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
- you want the replay minimap tooling and smooth interpolated motion

## Runtime Decision Path

`fw.main_with_airsim` reads `src/airsim/config/runtime_settings.json`.

Decision rules:

- `visualization.model_visualise=true` → run `run_visualise_models.py`
- `visualization.inference_visualise=true` → run `run_inference.py`
- if both replay and inference flags are true, replay wins

## `run_inference.py`

### Scope

`run_inference.py` is AirSim-only.

It is meant for PPO policies trained inside the UE5-backed `AirSimEnv`. It does not contain a framework-replay compatibility path or a separate direct-control mode.

### Effective control loop

1. load the PPO model
2. construct `AirSimEnv` in inference mode
3. call `env.reset()` at the start of each episode
4. predict an action from the current observation
5. call `env.step(action)`
6. let `AirSimEnv.step()` handle:
   - action validation and clipping to `[-1, 1]`
   - steering dynamics (low-pass filter + rate limiter)
   - control mapping to `CarControls`
   - control send to UE5
   - physics step (pause-stepping or sim-time-stepping)
   - state readback
   - 18-field observation update
   - reward calculation and normalization
   - checkpoint progress tracking
   - termination / truncation detection
7. stop the episode when `terminated` or `truncated` becomes true
8. call `env.close()` on shutdown

### Exports

Each run creates a timestamped session folder in `exports/`.

| File | Contents |
|------|---------|
| `metadata.json` | model path, runtime config, `performance_summary` block |
| `airsim_telemetry.csv` | UE5-side position, velocity, controls, timestamps (from bridge) |
| `inference_session.csv` | per-step: episode, x, y, heading, speed, reward, steering, throttle, checkpoint index, CTE, outcome |
| `trajectory_2d_vs_3d.png` | side-by-side 2D (framework) vs 3D (UE5) trajectory comparison, annotated with shape RMSE and DTW distance |

`performance_summary` in `metadata.json` includes:

- total reward, total duration
- cumulative 2D and 3D distance
- mean and max cross-track error
- per-episode outcome and metrics

### What this means for tasks

- `T-026` is handled through the native `AirSimEnv.step()` action → physics → feedback loop
- `T-027` is handled through `AirSimEnv.reset()`, `AirSimEnv.step()`, and `AirSimEnv.close()`

## `run_visualise_models.py`

### Scope

`run_visualise_models.py` is the modular framework-model replay runner.

Framework physics remain authoritative. UE5 is used only as renderer and mirror.

### Effective replay loop

1. resolve adapter from model family and environment name (registry in `visualisation_models/model_adapters.py`)
2. build framework env (`PySimEnv` or `FhSimEnv`)
3. load PPO policy
4. step framework env locally
5. extract motion state (x, y, yaw, vx, vy, yaw_rate) via `CoordTransform`
6. mirror pose/kinematics/object state into UE5 with smooth interpolation
7. update UE5 top-down minimap camera if enabled
8. export telemetry bundle

### Motion interpolation

Between framework steps, the runner interpolates motion to avoid discrete jumps in UE5.

| Setting | Description |
|---------|-------------|
| `render_interpolation_steps` | Number of sub-steps per framework step (runtime_settings.json default: 8) |
| `max_translation_step` | Max position delta per sub-step (adaptive clamp) |
| `max_yaw_step_deg` | Max yaw delta per sub-step (adaptive clamp) |
| `use_kinematics_in_ue5` | Use `simSetKinematics()` for smooth velocity-aware motion |
| `min_render_hz` | Minimum render update rate |

### Control animation

The runner can optionally animate the ship's steering and throttle visuals in UE5 independently of physics.

| Setting | Description |
|---------|-------------|
| `mirror_controls_in_ue5` | Pass raw controls through to UE5 |
| `animate_steering_in_ue5` | Animate steering wheel/rudder visuals |
| `animate_throttle_in_ue5` | Animate propulsion/throttle visuals |
| `steering_visual_gain` | Steering animation scaling (default 0.35) |
| `throttle_visual_gain` | Throttle animation scaling (default 0.892) |

### Supported adapters

Current adapter families:

- `pysim`
- `fhsim`

The adapter registry lives in `visualisation_models/model_adapters.py`. Each adapter implements `ModelVisualisationAdapter` and provides environment construction, pose/velocity extraction, and control animation.

### Exports

| File | Contents |
|------|---------|
| `metadata.json` | model info, runtime config, `performance_summary` block |
| `airsim_telemetry.csv` | UE5-side position, velocity, controls, timestamps |
| `visualisation_session.csv` | per-step: episode, framework state, controls, outcome |
| `trajectory_2d_vs_3d.png` | side-by-side 2D vs 3D trajectory comparison |

## Minimap

### UE5 camera minimap

`UE5TopDownMinimap` uses a top-down AirSim camera rendered inside UE5.

The camera is positioned above the current ship position and pointed straight down. When `ue5_camera_follow_heading=true`, the camera rotates to follow ship heading.

| Setting | Description |
|---------|-------------|
| `show_minimap` | Enable minimap (default false) |
| `minimap_update_hz` | Maximum camera update rate |
| `ue5_camera_name` | Name of the camera actor in UE5 |
| `ue5_camera_height_m` | Camera height above ship (default 120 m) |
| `ue5_camera_fov_deg` | Camera field of view |
| `ue5_camera_follow_heading` | Rotate camera with ship heading |

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

### Verbose diagnostics

Both runners default to concise console output.

Use `--verbose` when needed:

```bash
python -m airsim.inference.run_inference --model models/AirSimEnv_PPO_policy.zip --verbose
python -m airsim.inference.run_visualise_models --model models/PySimEnv_PPO_policy.zip --verbose
```

Or set environment-level verbose mode:

```bash
set FW_AIRSIM_VERBOSE=1
```

## Important Runtime Settings

### Connection

- `airsim.ip`
- `airsim.port`
- `airsim.save_slot`
- `airsim.simulation_speed`

### Replay selection

- `visualization.model_visualise`
- `visualization.model_family`
- `visualization.environment_name`

### Inference selection

- `visualization.inference_visualise`
- `visualization.inference_model_path`

### Replay rendering and minimap

- `visualization.show_minimap`
- `visualization.minimap_update_hz`
- `visualization.bridge_telemetry_hz`
- `visualization.render_interpolation_steps`
- `visualization.throttle_visual_gain`
- `visualization.steering_visual_gain`
- `visualization.use_kinematics_in_ue5`
- `visualization.vehicle_object_name`
- `visualization.ue5_camera_name`
- `visualization.ue5_camera_height_m`
- `visualization.ue5_camera_follow_heading`
- `visualization.draw_ship_trail_in_ue5`
- `visualization.ship_trail_min_segment_m`

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

### Framework-trained model passed to `run_inference.py`

- switch to `run_visualise_models.py`
- `run_inference.py` is reserved for models trained in `AirSimEnv`

### `plot_airsim_path.py`

This tool exists as a standalone telemetry plotter. It is not wired into the main runtime paths automatically.
