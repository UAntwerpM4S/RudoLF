# AirSim Inference and Visualisation Runbook

This folder now has two primary runtime entrypoints with different purposes:

- `run_inference.py`: UE5-driven policy inference using `AirSimEnv`.
- `run_visualise_models.py`: framework-driven replay mirrored into UE5.

The old file `visualisation_models/run_visualise_fossen.py` is kept as a legacy compatibility implementation/wrapper path. New runtime dispatch goes directly through `run_visualise_models.py`.

## Which Runner Should I Use?

Use `run_inference.py` when:

- UE5 (`AirSimEnv`) is your authoritative simulator.
- You want to execute a policy directly against live UE5 state and control.
- You are validating UE5-trained checkpoints or running Fossen-trained checkpoints through compatibility mapping in `AirSimEnv`.

Use `run_visualise_models.py` when:

- Framework physics (`PySimEnv` or `FhSimEnv`) must stay authoritative.
- UE5 should act as a visual renderer for framework-generated motion.
- You want minimap options (matplotlib or UE5 top-down camera).

## Runtime Flow

### 1) Main dispatcher

`fw.main_with_airsim` reads `airsim/config/runtime_settings.json` and checks:

- `visualization.fossen_visualise`
- `visualization.inference_visualise`

If `fossen_visualise=true`:

- `fw.main_with_airsim` calls `airsim.inference.run_visualise_models.run_visualise_models(...)`.

If `inference_visualise=true`:

- `fw.main_with_airsim` calls `airsim.inference.run_inference.run_inference(...)`.

If both are true:

- Fossen/model visualisation is prioritized and inference visualisation is skipped.

### 2) Model visualisation path (`run_visualise_models.py`)

- Loads PPO model.
- Selects framework environment family (`PySimEnv` or `FhSimEnv`, or explicit `model_family`).
- Steps framework environment.
- Mirrors state into UE5 through `AirSimBridge` pose/kinematics/object APIs.
- Optionally displays:
  - matplotlib minimap (`VisualiseModelsMinimap`) or
  - UE5 camera minimap (`UE5TopDownMinimap`) using camera pose/FOV/image APIs.

### 3) Inference path (`run_inference.py`)

- Loads PPO model.
- Connects `AirSimEnv` to UE5.
- Auto-detects model origin from filename (or via `--ue5` flag):
  - Fossen-compatible mode: uses compatibility observation/control path.
  - UE5-direct mode: maps policy action directly to UE5 control.
- Runs episode loop directly on UE5-backed environment state.

## Key Runtime Settings

The control file is `airsim/config/runtime_settings.json`.

### AirSim connection

- `airsim.ip`
- `airsim.port`
- `airsim.save_slot`
- `airsim.simulation_speed`

### Common visualisation control

- `visualization.max_steps`
- `visualization.episodes`
- `visualization.device`
- `visualization.log_every`

### Model visualisation selector

- `visualization.fossen_visualise` (true enables `run_visualise_models` path)
- `visualization.fossen_model_path`
- `visualization.model_family` (`auto`, `fossen`, `fhsim` depending on your naming conventions)
- `visualization.fossen_environment` (`auto`, `PySimEnv`, `FhSimEnv`)

### Inference selector

- `visualization.inference_visualise` (true enables `run_inference` path)
- `visualization.inference_model_path`

### UE5 mirror/minimap behavior (model visualisation path)

- `visualization.draw_calibration_reference`
- `visualization.mirror_controls_in_ue5`
- `visualization.animate_steering_in_ue5`
- `visualization.animate_throttle_in_ue5`
- `visualization.render_interpolation_steps`
- `visualization.steering_visual_gain`
- `visualization.throttle_visual_gain`
- `visualization.use_kinematics_in_ue5`
- `visualization.min_render_hz`
- `visualization.max_translation_step`
- `visualization.max_yaw_step_deg`
- `visualization.lock_path_in_ue5`
- `visualization.vehicle_object_name`
- `visualization.show_minimap`
- `visualization.use_ue5_camera_minimap`
- `visualization.ue5_camera_name`
- `visualization.ue5_camera_fov_deg`
- `visualization.ue5_camera_height_m`
- `visualization.ue5_camera_follow_heading`
- `visualization.ue5_preview_window`

## How to Run

From `src` folder:

### A) Framework replay mirrored into UE5

```bash
python -m airsim.inference.run_visualise_models --model models/PySimEnv_PPO_policy.zip
```

### B) UE5-driven policy inference

```bash
python -m airsim.inference.run_inference --model models/AirSimEnv_PPO_policy.zip
```

### C) Through unified runtime dispatcher

```bash
python -m fw.main_with_airsim
```

This route uses `runtime_settings.json` flags to choose between the two modes.

## Legacy Compatibility

`visualisation_models/run_visualise_fossen.py` remains present so existing scripts/imports can still work. New code should target:

- `airsim.inference.run_visualise_models`

## Troubleshooting

- UE5 unreachable:
  - Ensure UE5 is running in Play mode.
  - Verify `airsim.ip` and `airsim.port`.
- Wrong runner starts:
  - Check `visualization.fossen_visualise` and `visualization.inference_visualise`.
  - Remember: when both are true, `fossen_visualise` wins.
- Minimap not visible:
  - Set `visualization.show_minimap=true`.
  - For camera minimap, also set `visualization.use_ue5_camera_minimap=true` and verify camera name/FOV.
- Model family mismatch in replay:
  - Set `visualization.model_family` and `visualization.fossen_environment` explicitly instead of `auto`.
