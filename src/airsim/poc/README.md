# POC — Proof of Concept (Frozen Snapshots)

This folder contains **frozen copies** of the original AirSim bridge and
training scripts as they existed before the restructuring. They serve as
a reference point for the initial proof of concept and should **not** be
modified.

## Contents

| File                   | Original location                    | Description                                                                 |
| ---------------------- | ------------------------------------ | --------------------------------------------------------------------------- |
| `airsim_bridge_PID.py` | `fw/simulators/airsim_bridge_PID.py` | PID heading-tracking bridge that maps Fossen outputs to AirSim CarControls. |
| `train_airsim.py`      | `fw/air_sim/train_airsim.py`         | Entry point for RL training inside UE5 via CosysAirSim.                     |

## Why keep these?

- **Traceability**: shows the exact starting point of the AirSim integration.
- **Comparison**: useful for diffing against the cleaned-up versions in
  `airsim/communication/` and `airsim/train/`.
- **Rollback**: if the restructured code introduces regressions, these
  originals can be quickly substituted.

> **Note:** The active, maintained code lives in the parent `airsim/`
> package — not here.
