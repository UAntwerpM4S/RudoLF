# AirSim Integration Package

> **CosysAirSim bridge, Gymnasium RL environment, coordinate calibration, and analysis tools for training autonomous navigation agents in Unreal Engine 5.**

---

## Overview

The `airsim` package connects the RudoLF reinforcement-learning framework to [CosysAirSim](https://github.com/Cosys-Lab/Cosys-AirSim) — a photorealistic vehicle simulator built on Unreal Engine 5. It enables:

- **RL training inside UE5** — the `AirSimEnv` Gymnasium environment reads vehicle state (position, heading, speed) directly from UE5 and sends steering / throttle commands back, replacing the Fossen mathematical model with UE5's physics engine.
- **Simultaneous path following** — the `AirSimBridge` PID adapter runs alongside the Fossen simulation, mirroring the trained policy's ship in UE5 in real time.
- **Coordinate calibration** — an interactive wizard aligns the Fossen (east, north) frame with AirSim's (NED) frame using two reference points, automatically handling scale, rotation, axis mirroring, and origin drift.

### Architecture

```
┌─────────────────┐         ┌──────────────────┐         ┌──────────────────┐
│  RL Agent (PPO)  │────────▶│  Fossen Ship Sim  │────────▶│    Matplotlib     │
│   (Stable B3)    │  action │  (Python, NED)    │  state  │   Live Animation  │
└─────────────────┘         └────────┬─────────┘         └──────────────────┘
                                     │ ship_state
                                     ▼
                            ┌──────────────────┐         ┌──────────────────┐
                            │  AirSimBridge     │────────▶│   Unreal Engine 5 │
                            │  (PID + XTE)      │  steer  │   CosysAirSim     │
                            └──────────────────┘  + thr   └──────────────────┘
```

The bridge does **not** forward raw RL actions to UE5. Instead, it reads the _outputs_ of the Fossen model (heading + speed) and uses PID controllers to make the AirSim vehicle match them. This decouples the control logic from differences in physics engines.

---

## Package Structure

```
airsim/
├── __init__.py                   # Re-exports CoordTransform, get_airsim_config
│
├── config/                       # Configuration & calibration data
│   ├── airsim_config.py          # PID gains, network settings, telemetry flags
│   ├── calibration.json          # Fossen ↔ AirSim coordinate calibration
│   └── settings_training.json    # UE5 simulation mode / speed settings
│
├── communication/                # UE5 ↔ Fossen communication layer
│   ├── airsim_bridge.py          # PID heading-tracking adapter
│   └── coord_transform.py       # Bidirectional coordinate mapping
│
├── environments/                 # Gymnasium RL environments
│   └── airsim_env.py             # AirSimEnv — drop-in replacement for PySimEnv
│
├── train/                        # Training scripts
│   └── train_airsim.py           # Entry point for RL training inside UE5
│
├── inference/                    # Visualization & analysis scripts
│   ├── plot_airsim_path.py       # Live dead-reckoning path plotter
│   └── verify_headless_plot.py   # CI-friendly headless plot export
│
├── tools/                        # Calibration & telemetry utilities
│   ├── calibrate_coords.py       # Interactive Fossen↔AirSim calibration wizard
│   └── analyze_telemetry.py      # Offline telemetry statistics
│
├── poc/                          # Frozen proof-of-concept snapshots
│   ├── README.md                 # Documents the POC scope
│   ├── airsim_bridge_PID.py      # Original bridge (unmodified)
│   └── train_airsim.py           # Original training script (unmodified)
│
└── tests/                        # Pytest unit tests
    ├── conftest.py               # Mock AirSim fixtures
    ├── test_airsim_bridge.py     # Bridge helper & init tests
    ├── test_airsim_config.py     # Config key & default tests
    └── test_coord_transform.py   # Round-trip & heading transform tests
```

---

## Quick Start

### Prerequisites

| Requirement        | Version | Notes                                     |
| ------------------ | ------- | ----------------------------------------- |
| Python             | ≥ 3.12  |                                           |
| CosysAirSim        | latest  | `pip install cosysairsim`                 |
| Unreal Engine 5    | ≥ 5.3   | With the CosysAirSim plugin + water level |
| Stable Baselines 3 | ≥ 2.0   | `pip install stable-baselines3`           |
| pytest             | ≥ 8.0   | `pip install pytest` (for tests only)     |

### 1 — Calibrate coordinates (one-time)

```bash
cd src
python -m airsim.tools.calibrate_coords
```

Follow the prompts to drive the vehicle to two reference points. The wizard writes `calibration.json` and draws the Fossen polygon overlay in UE5 for visual verification.

### 2 — Train inside UE5

```bash
cd src
python -m airsim.train.train_airsim
```

This connects to a running UE5 + CosysAirSim instance, creates an `AirSimEnv` Gymnasium environment, and trains a PPO agent using Stable Baselines 3.

### 3 — Visualise via the bridge

To run a trained policy in the Fossen sim **and** mirror it in UE5 simultaneously:

```bash
cd src
python -m fw.main
```

Set `use_airsim: True` in `airsim/config/airsim_config.py`. The bridge automatically connects and starts forwarding.

### 4 — Analyse telemetry

```bash
cd src
python -m airsim.tools.analyze_telemetry
```

### 5 — Run tests

```bash
cd src
python -m pytest airsim/tests -v
```

---

## Key Components

### `AirSimBridge` — Heading-Tracking Adapter

**Module:** `airsim.communication.airsim_bridge`

The bridge connects to AirSim via RPC and, on every simulation step:

1. Reads the Fossen ship's heading, speed, and yaw rate.
2. Converts the Fossen heading to AirSim yaw: `AirSim_yaw = π/2 − heading`.
3. Reads the AirSim vehicle's actual yaw from its pose quaternion.
4. Computes a **steering command** using:
   - **PID** on heading error (`Kp`, `Ki`, `Kd`).
   - **Feedforward** on yaw rate (`Kff`).
   - **Cross-track error correction** (`lat_gain`).
5. Computes a **throttle command** using:
   - **PI** on speed error (`throttle_kp`, `throttle_ki`).
   - **Cumulative-distance correction** (`dist_gain`, `dist_ki`).
6. Sends `CarControls(steering, throttle)` to AirSim.

All parameters are configurable in `airsim/config/airsim_config.py`.

### `CoordTransform` — Bidirectional Coordinate Mapping

**Module:** `airsim.communication.coord_transform`

Handles the complex mapping between the Fossen (x = east, y = north) and AirSim (x = north, y = east, NED) coordinate frames. Supports:

- **Scale** — different distance units between frames.
- **Rotation** — arbitrary rotation angle.
- **Axis mirroring** — independent X / Y flips.
- **Translation** — different origin positions.
- **Heading ↔ Yaw** conversion with correct mirroring.
- **Origin-drift auto-correction** — compensates for UE5 PlayerStart shifts.

Configured from `calibration.json` (generated by `calibrate_coords.py`).

### `AirSimEnv` — Gymnasium RL Environment

**Module:** `airsim.environments.airsim_env`

A drop-in replacement for `PySimEnv` that uses UE5's physics engine instead of the Fossen mathematical model:

- **Same observation space** (10-dim: velocities, distances, heading errors).
- **Same action space** (`[steering, throttle]` in [-1, 1]).
- **Same reward function** (forward progress + heading alignment + cross-track penalty + rudder penalty).
- **Same termination conditions** (collision, out-of-water, excessive heading change, cross-track limit).

Additional UE5-specific features:

- **Deterministic frame stepping** — pauses UE5 and advances frame-by-frame for reproducible training.
- **Training mode** — disables UE5 rendering, fog, particles, shadows to maximise training speed.
- **Drift auto-correction** — detects and compensates for UE5 coordinate-origin shifts.

---

## Configuration Reference

All AirSim parameters live in `airsim/config/airsim_config.py`:

| Parameter              | Default | Description                                      |
| ---------------------- | ------- | ------------------------------------------------ |
| `use_airsim`           | `True`  | Enable / disable the AirSim bridge               |
| `airsim_ip`            | `""`    | AirSim server IP (empty = localhost)             |
| `airsim_port`          | `41451` | AirSim RPC port                                  |
| `airsim_kp`            | `5.0`   | Heading PID — proportional gain                  |
| `airsim_ki`            | `0.5`   | Heading PID — integral gain                      |
| `airsim_kd`            | `1.0`   | Heading PID — derivative gain                    |
| `airsim_ff`            | `0.5`   | Heading feedforward gain (yaw rate anticipation) |
| `airsim_lat_gain`      | `0.5`   | Cross-track error correction gain                |
| `airsim_speed_scale`   | `1.05`  | Speed multiplier (Fossen → AirSim)               |
| `airsim_throttle_kp`   | `1.0`   | Throttle PI — proportional gain                  |
| `airsim_throttle_ki`   | `0.2`   | Throttle PI — integral gain                      |
| `airsim_dist_gain`     | `0.1`   | Cumulative distance error — proportional gain    |
| `airsim_dist_ki`       | `0.01`  | Cumulative distance error — integral gain        |
| `airsim_log_telemetry` | `True`  | Write per-step CSV telemetry                     |

---

## Coordinate Frames

| Frame  | X-axis | Y-axis | Heading reference | Positive direction |
| ------ | ------ | ------ | ----------------- | ------------------ |
| Fossen | East   | North  | From East         | Counter-clockwise  |
| AirSim | North  | East   | From North (NED)  | Clockwise          |

**Conversion:** `AirSim_yaw = π/2 − fossen_heading`

The `CoordTransform` class handles all conversions, including scale, rotation, and axis mirroring as specified in `calibration.json`.

---

## Telemetry

When `airsim_log_telemetry` is enabled, the bridge writes a CSV file with the following columns per simulation step:

| Column              | Unit    | Description                     |
| ------------------- | ------- | ------------------------------- |
| `step`              | —       | Simulation step counter         |
| `ship_speed`        | m/s     | Fossen ship surge speed         |
| `car_speed`         | m/s     | AirSim vehicle speed            |
| `target_speed`      | m/s     | Desired speed after corrections |
| `dist_error`        | m       | Cumulative distance error       |
| `heading_error_deg` | degrees | Heading error (target − actual) |
| `ship_heading_deg`  | degrees | Fossen ship heading             |
| `car_yaw_deg`       | degrees | AirSim vehicle yaw              |
| `steer`             | [-1, 1] | Commanded steering              |
| `throttle`          | [0, 1]  | Commanded throttle              |
| `ship_yaw_rate`     | rad/s   | Fossen ship yaw rate            |

Use `python -m airsim.tools.analyze_telemetry` to compute accuracy statistics, or `python -m airsim.inference.plot_airsim_path` for live visualisation.

---

## Tests

```bash
cd src
python -m pytest airsim/tests -v
```

The test suite covers:

- **`test_coord_transform.py`** — round-trip position/heading conversions, mirroring, JSON loading.
- **`test_airsim_config.py`** — config key presence, default value sanity, immutability.
- **`test_airsim_bridge.py`** — `_normalize_angle`, `_yaw_from_quaternion`, graceful disconnected state.

Tests run without UE5 or CosysAirSim (mock fixtures in `conftest.py`).
