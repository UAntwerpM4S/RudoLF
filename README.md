# RudoLF

Multi-paradigm Reinforcement Learning Framework

![RudoLF Logo](docs/logo.png)

> The following project is composed of a Gazebo simulator and two Python simulators (discrete and continuous env) used to train an agent for autonomous navigation. The trained policy can be visualized in both **Matplotlib** and **Unreal Engine 5** (via CosysAirSim) simultaneously.

---

## UE5 Visualization — CosysAirSim Bridge (WIP)

RudoLF can now drive a vehicle inside **Unreal Engine 5** in real-time (TESTING PHASE!!!), mirroring the RL agent's behavior from the Fossen ship simulation. The bridge translates RL actions (`[rudder, thrust]`) into AirSim `CarControls` (`[steering, throttle]`), achieving synchronized path following between both environments.

![RudoLF UE5 vs Matplotlib](docs/rudolf_ue5_vs_matplotlib.png)

### Architecture

```
┌─────────────────┐         ┌──────────────────┐         ┌──────────────────┐
│  RL Agent (PPO)  │────────▶│  Fossen Ship Sim  │────────▶│    Matplotlib     │
│   (Stable B3)    │  action │  (Python, NED)    │  state  │   Live Animation  │
└─────────────────┘         └────────┬─────────┘         └──────────────────┘
                                     │ ship_state
                                     ▼
                            ┌──────────────────┐         ┌──────────────────┐
                            │  AirSim Bridge    │────────▶│   Unreal Engine 5 │
                            │  (PID + XTE)      │  steer  │   CosysAirSim     │
                            └──────────────────┘  + thr   └──────────────────┘
```

### Key Components

| Component         | File                                 | Purpose                                           |
| ----------------- | ------------------------------------ | ------------------------------------------------- |
| **AirSim Bridge** | `src/fw/simulators/airsim_bridge.py` | PID+FF steering & PI throttle controller          |
| **Configuration** | `src/fw/config.py`                   | All tuning parameters (`airsim_*`)                |
| **Path Plotter**  | `src/plot_airsim_path.py`            | Dead-reckoning path reconstruction from telemetry |
| **Telemetry Log** | `src/airsim_telemetry.csv`           | Per-step speed, heading, steering, and error data |

### Control Strategy

- **Steering**: PID + Feedforward + Cross-Track Error correction
  - `Kp` heading error + `Ki` integral + `Kd` derivative + `Kff` yaw rate + XTE lateral correction
- **Throttle**: PI controller tracking cumulative path distance (arc length)
  - Cornering logic reduces speed during large heading deviations

### Path Reconstruction

Telemetry data can be visualized with `plot_airsim_path.py`, which reconstructs both paths via dead reckoning:

![AirSim Path Reconstruction](docs/airsim_path_reconstruction.png)

### Quick Start

1. Ensure **CosysAirSim** is running with the UE5 environment.
2. In `src/fw/config.py`, set:
   ```python
   'use_airsim': True,
   ```
3. Run the simulation:
   ```bash
   cd src
   python -m fw.main
   ```
4. _(Optional)_ Visualize telemetry in real-time:
   ```bash
   python -m plot_airsim_path
   ```

### Configuration Reference

| Parameter              | Default | Description                                              |
| ---------------------- | ------- | -------------------------------------------------------- |
| `airsim_kp`            | `5.0`   | Proportional gain — steering response to heading error   |
| `airsim_ki`            | `0.5`   | Integral gain — eliminates steady-state heading drift    |
| `airsim_kd`            | `1.0`   | Derivative gain — dampens steering oscillation           |
| `airsim_ff`            | `0.5`   | Feedforward gain — proactive steering from ship yaw rate |
| `airsim_lat_gain`      | `0.5`   | Cross-Track Error gain — corrects lateral position drift |
| `airsim_speed_scale`   | `1.06`  | Speed multiplier to match simulation units               |
| `airsim_log_telemetry` | `True`  | Logs step-by-step control data to CSV                    |

---

## ROS2 and Gazebo

- ROS2 Humble
- Gazebo Garden

```bash
source path/to/ros/install/setup.bash
cd path/to/vrx/project
./run.sh
source install/setup.bash
ros2 launch competition_custom.launch.py world:=sydney_regatta_custom
```

## Python Simulator

- Python 3.12
- Numpy, Matplotlib, Stable Baselines 3

```bash
cd src
python -m fw.main
```
