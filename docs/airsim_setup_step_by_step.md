# AirSim Step-By-Step Setup Guide

This guide describes the current project-specific setup for running RudoLF with
UE5 and CosysAirSim.

It covers:

1. creating and activating `.venv`
2. installing Python dependencies
3. installing `cosysairsim`
4. replacing the installed `cosysairsim` package with the bespoke project
   version used by the UE5 AirSim development workflow
5. preparing UE5 + the RudoLF AirSim project
6. validating the setup

## Source Of Truth For What Actually Runs

The runtime decision path starts in code, not in the setup guide.

The current source of truth is:

1. `src/fw/config.py`
2. `src/fw/main.py`
3. `src/fw/main_with_airsim.py`
4. `src/airsim/config/runtime_settings.json`

What this means in practice:

- `fw.config.CONFIG_PARAMETERS["env_type"]` decides which simulator(s) are selected
- `fw.config.CONFIG_PARAMETERS["action"]` decides whether the framework asks for `train` or `visualize`
- `fw.main` executes the configured environments in order
- whenever `AirSimEnv` appears in that ordered list, `fw.main` delegates that step to `fw.main_with_airsim`
- once the AirSim step is selected, `runtime_settings.json` refines how the AirSim step behaves

So:

- `fw.config` selects the simulator and the high-level mode
- `runtime_settings.json` selects the AirSim-specific sub-mode and AirSim-specific runtime parameters

## Current Checked-In Defaults

At the time this document was updated, the checked-in source defaults are:

- in `src/fw/config.py`
  - `env_type = ["PySimEnv", "AirSimEnv"]`
  - `action = "train"`
- in `src/airsim/config/runtime_settings.json`
  - `visualization.model_visualise = true`
  - `visualization.inference_visualise = false`
  - `training.speed_profile = "turbo"`

That means the repository, as currently checked in, is biased toward framework
training with an ordered `PySimEnv -> AirSimEnv` flow. If you switch the
framework action to `visualize`, the AirSim runtime defaults are biased toward
model visualisation (`model_visualise = true`).

## How Train vs Visualize Is Actually Resolved

### `fw.main`

`fw.main` reads `src/fw/config.py`.

Examples:

- if `env_type = ["PySimEnv"]` and `action = "train"`, the generic framework training path runs
- if `env_type = ["AirSimEnv"]` and `action = "train"`, `fw.main` delegates the AirSim step to `fw.main_with_airsim`
- if `env_type = ["PySimEnv", "AirSimEnv", "PySimEnv"]`, `fw.main` preserves that exact order and delegates only the AirSim step to `fw.main_with_airsim`

### `fw.main_with_airsim`

Once `fw.main_with_airsim` is entered:

- if the selected action is `train`, it builds `AirSimEnv` and runs trainer-based AirSim training
- if the selected action is `visualize`, it first checks the AirSim runtime visualization flags

Inside `src/airsim/config/runtime_settings.json`:

- `visualization.model_visualise = true` runs `airsim.inference.run_visualise_models`
- `visualization.inference_visualise = true` runs `airsim.inference.run_inference`
- if both are true, model visualisation wins

If neither AirSim visualisation flag is enabled, `fw.main_with_airsim` falls back
to the generic trainer visualisation path for the selected environment.

## How To Intentionally Choose A Runtime Mode

### Framework training

Use this when you want standard framework training behavior:

- set `src/fw/config.py`
  - `env_type` to the simulator(s) you want
  - `action = "train"`

### AirSimEnv training

Use this when you want UE5-backed training:

- set `src/fw/config.py`
  - `env_type = ["AirSimEnv"]`
  - `action = "train"`
- then tune AirSim-specific training behavior in `src/airsim/config/runtime_settings.json`
  - `training.speed_profile`
  - `training.simulation_speed`
  - `training.frames_per_step`
  - `training.time_step`

### Modular framework-model replay in UE5

Use this when you want `PySimEnv` or `FhSimEnv` to remain authoritative while
UE5 mirrors the motion:

- set `src/fw/config.py`
  - `env_type = ["AirSimEnv"]`
  - `action = "visualize"`
- set `src/airsim/config/runtime_settings.json`
  - `visualization.model_visualise = true`
  - `visualization.inference_visualise = false`
  - choose `visualization.model_path`
  - choose `visualization.model_family`
  - choose `visualization.environment_name`

### AirSimEnv inference in UE5

Use this when the model itself was trained inside `AirSimEnv`:

- set `src/fw/config.py`
  - `env_type = ["AirSimEnv"]`
  - `action = "visualize"`
- set `src/airsim/config/runtime_settings.json`
  - `visualization.model_visualise = false`
  - `visualization.inference_visualise = true`
  - choose `visualization.inference_model_path`

## Prerequisites

- Windows machine with Unreal Engine 5 installed
- access to the RudoLF repository
- access to the UE5 AirSim development Google Drive that contains the bespoke
  `cosysairsim` package folder
- a Python version compatible with the project virtual environment

## 1. Create And Activate `.venv`

From the repository root:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks script execution:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

## 2. Install The Python Dependencies

Upgrade packaging tools first:

```powershell
python -m pip install --upgrade pip setuptools wheel
```

Then install the project dependencies as normally used in your local workflow.
Make sure to install packages from requirements.txt!

## 3. Install The Standard `cosysairsim` Package

Install the public package into the active virtual environment:
"Cosys Air Sim package should be installed if you ran pip install -r requirements.txt in your venv."

```powershell
python -m pip install cosysairsim
```

This gives you the baseline package structure inside:

```text
.venv/Lib/site-packages/cosysairsim/
```

## 4. Replace It With The Bespoke `cosysairsim` Folder

Important: the standard public `cosysairsim` package is not enough for this
project.

RudoLF depends on newer project-specific RPC calls that are available in the
bespoke UE5 AirSim development package but not in the standard published
package.

After installing `cosysairsim`, do the following:

1. Download the bespoke `cosysairsim` folder from the Google Drive shared by
   the UE5 AirSim development workflow.
2. Open your virtual environment package directory:

```text
.venv/Lib/site-packages/
```

3. Replace the installed `cosysairsim` folder with the downloaded bespoke
   `cosysairsim` folder.

In other words:

- remove or rename `.venv/Lib/site-packages/cosysairsim`
- copy the downloaded bespoke `cosysairsim` folder into that same location

After replacement, the effective import path should still be:

```text
.venv/Lib/site-packages/cosysairsim/
```

but its contents must be the bespoke project version, not the default pip
version.

## 5. Prepare UE5 + RudoLF

Use the UE5 project and AirSim development build that belongs to the RudoLF
workflow.

Expected high-level setup:

- UE5 project opens successfully
- the CosysAirSim integration is enabled in that project
- the RudoLF-compatible world/save-slot data is available
- UE5 can enter Play mode
- the AirSim RPC server becomes reachable on the configured port

The default runtime settings in this repository expect:

- IP: `127.0.0.1`
- port: `41451`
- save slot: `simulator_1`

These can be changed in:

```text
src/airsim/config/runtime_settings.json
```

## 6. Validate The Python Side

With `.venv` active, check that the bespoke package imports:

```powershell
python -c "import cosysairsim; print(cosysairsim.__file__)"
```

The printed path should point inside:

```text
.venv\Lib\site-packages\cosysairsim\
```

and must be the replaced bespoke package contents.

## 7. Validate The UE5 Connection

Start UE5, enter Play mode, then run:

```powershell
cd src
python -m pytest tests/airsim/communication/test_bridge.py -q
```

or a lighter manual runtime check:

```powershell
cd src
python -m fw.main_with_airsim
```

depending on the runtime settings currently selected.

## 8. Optional Calibration Step

If calibration data is missing or needs to be redrawn:

```powershell
cd src
python -m fw.common.airsim.tools.calibrate_coords
```

This prepares the transform used between framework coordinates and AirSim/UE5
coordinates.

## 9. Common Runtime Paths

These commands only describe the entrypoint. The actual behavior still depends
on `src/fw/config.py` and `src/airsim/config/runtime_settings.json` as explained
above.

### AirSimEnv training

```powershell
cd src
python -m fw.main
```

### AirSimEnv inference

```powershell
cd src
python -m airsim.inference.run_inference --model models/AirSimEnv_PPO_policy.zip
```

### Framework-model replay in UE5

```powershell
cd src
python -m airsim.inference.run_visualise_models --model models/PySimEnv_PPO_policy.zip
```

## Troubleshooting

### `cosysairsim` imports, but RPC calls are missing

Most likely cause:

- the public pip package is still installed
- the bespoke Google Drive package folder did not fully replace it

Fix:

- re-copy the bespoke `cosysairsim` folder into `.venv/Lib/site-packages/`

### UE5 starts, but RudoLF does not connect

Check:

- UE5 is in Play mode
- RPC port matches `runtime_settings.json`
- local firewall is not blocking the port

### A visualisation starts in the wrong session state

The bridge now includes fallback handling for unreliable save-slot reloads, but
you should still verify:

- the requested save slot exists
- UE5 had enough time to finish reloading
- the bespoke `cosysairsim` package is installed

## Related Docs

- [AirSim Package Reference](../src/airsim/README.md)
- [AirSim Architecture](airsim_architecture.md)
- [AirSim Inference and Visualisation Runbook](../src/airsim/inference/README.md)
