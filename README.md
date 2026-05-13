# RudoLF

Multi-paradigm reinforcement learning framework for autonomous ship navigation across framework simulators and UE5/CosysAirSim.

![RudoLF Logo](docs/logo.png)

## Framework Essentials

RudoLF provides a single primary entrypoint:

```bash
cd src
python -m fw.main
```

`fw.main` behavior is controlled by `src/fw/config.py`:

- `env_type`: a single environment or an ordered list, for example `['PySimEnv', 'AirSimEnv', 'PySimEnv']`.
- `action`: `train` or `visualize`.

### Ordered Multi-Environment Execution

- `fw.main` executes environments in the configured order.
- AirSim entries are delegated to `fw.main_with_airsim` at their exact position in the list.
- If an AirSim step cannot run (UE5 unavailable), that step is skipped and the run continues with remaining configured environments.
- Repeated environments in `env_type` are preserved and executed as configured.

### Runtime Routing

- Non-AirSim environments run through the generic framework trainer/visualizer flow.
- AirSim-specific runtime behavior (visualization mode, training speed profile, save slot, connection settings) is configured in `src/airsim/config/runtime_settings.json`.
- AirSim visualisation and inference exports are written to the workspace-level `exports/` folder.

### AirSim Training

`AirSimEnv` is the Gymnasium-compatible environment backed by UE5/CosysAirSim physics.

The policy receives an 18-field route-relative observation vector in the vehicle-fixed frame and outputs continuous steering and throttle actions. Steering dynamics (low-pass filter + rate limiter) are applied before controls are sent to UE5.

Training speed is configured via `training.speed_profile` in `runtime_settings.json`:

| Profile  | sim speed | Pause stepping | Recommended use        |
|----------|-----------|----------------|------------------------|
| `safe`   | 20×       | yes            | New training (default) |
| `fast`   | 40×       | yes            | Faster iteration       |
| `turbo`  | 80×       | yes            | Maximum throughput     |
| `normal` | 1×        | no             | Debug / real-time      |

`StrictSuccessStoppingCallback` halts training automatically once a configurable number of consecutive strict-success episodes is reached.

A checkpoint-based curriculum is available via `curriculum_enabled=true`. When active, each episode starts from a randomly chosen sub-route window that expands automatically as the agent achieves consecutive successes, progressively exposing the full route. A soft offtrack-limit decay also runs in parallel to ease early exploration.

Transfer learning from a pretrained 2D policy (PySimEnv or FhSimEnv) is supported via the `transfer_learning` block in `runtime_settings.json`. The first linear layer acts as a trainable mapping layer for the extra AirSim observation dimensions; deeper layers are transferred and optionally frozen.

### AirSim Exports

Each AirSim inference or visualization session creates its own timestamped export folder under `exports/`.

| File                                                   | Contents                                                      |
|--------------------------------------------------------|---------------------------------------------------------------|
| `inference_session.csv` or `visualisation_session.csv` | session-level telemetry                                       |
| `airsim_telemetry.csv`                                 | bridge-side AirSim state snapshots                            |
| `metadata.json`                                        | model, environment, runtime config, and `performance_summary` |
| `trajectory_2d_vs_3d.png`                              | side-by-side 2D (framework) vs 3D (UE5) trajectory comparison |

`metadata.json` includes a `performance_summary` section with total reward,
duration, 2D/3D distance, and mean/max cross-track error.

## Quick Start

### Framework run

```bash
cd src
python -m fw.main
```

### AirSim-only run

```bash
cd src
python -m fw.main_with_airsim
```

### Live AirSimEnv inference in UE5

```bash
cd src
python -m airsim.inference.run_inference --model models/AirSimEnv_PPO_policy.zip
```

### Framework replay in UE5

```bash
cd src
python -m airsim.inference.run_visualise_models --model models/PySimEnv_PPO_policy.zip
```

### Common AirSim utilities

```bash
cd src
python -m fw.common.airsim.tools.calibrate_coords
python -m fw.common.airsim.tools.analyze_telemetry
```

## AirSim Documentation

For full AirSim technical documentation — installation, configuration, training, visualization, inference, architecture, and testing — see:

- [AirSim Integration Manual](docs/airsim.md)

## Testing

Run all tests:

```bash
python -m pytest -v
```

Run AirSim-focused tests:

```bash
python -m pytest tests/airsim -v
```

Run docstring enforcement:

```bash
python -m pytest tests/fw/test_docstring_enforcement.py -v
```

Run framework routing tests:

```bash
python -m pytest tests/fw -v
```

## ROS2 and Gazebo (separate stack)

- ROS2 Humble
- Gazebo Garden

```bash
source path/to/ros/install/setup.bash
cd path/to/vrx/project
./run.sh
source install/setup.bash
ros2 launch competition_custom.launch.py world:=sydney_regatta_custom
```
