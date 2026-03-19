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
- If an AirSim step cannot run (for example UE5 unavailable), that AirSim step is skipped and the run continues with remaining configured environments.
- Repeated environments in `env_type` are preserved and executed as configured.

### Runtime Routing

- Non-AirSim environments run through the generic framework trainer/visualizer flow.
- AirSim-specific runtime behavior (UE5 visualisation mode, training speed profile, save slot, connection settings) is configured in `src/airsim/config/runtime_settings.json`.

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

### Common AirSim utilities

```bash
cd src
python -m fw.common.airsim.tools.calibrate_coords
python -m fw.common.airsim.tools.analyze_telemetry
```

## AirSim Docs Hub

For full AirSim technical documentation, use:

- [AirSim Package Reference](src/airsim/README.md)
- [AirSim Inference and Visualisation Runbook](src/airsim/inference/README.md)
- [AirSim Workflow Rationale](src/airsim/verantwoording_workflow.md)
- [AirSim Architecture](docs/airsim_architecture.md)

## Testing

Run all tests:

```bash
python -m pytest -v
```

Run AirSim-focused tests:

```bash
python -m pytest tests/airsim -v
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
