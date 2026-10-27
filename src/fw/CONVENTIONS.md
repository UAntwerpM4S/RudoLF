# RudoLF conventions

One convention for the whole framework: `ShipSimulator.integrate`, every dynamics model, the surrogate datasets and checkpoints, and FH's own body frame. Sign conversions happen **only** at the places listed under *Conversion points*. Anywhere else, a sign flip is a bug.

`fw/surrogate/validation/check_model_conventions.py` runs every dynamics model, FH included, with the same commands and checks turning direction, yaw-rate and sway signs, and the thrust response. `tests/surrogate/test_conventions.py` covers the framework side (`python -m pytest tests/surrogate`, run from the repository root). The FH side was measured with `check_sway_sign.py`, `check_rudder_sign.py`, `check_initial_rudder_sign.py` and `check_model_conventions.py` (all in `fw/surrogate/validation/`) on 2026-10-19. Every model passed for the Tripoli and the Myzako 37, apart from the UA surrogates, which were not yet retrained at the time; they were retrained afterwards (the v2 checkpoints in `fw/checkpoints/`) and validated against FH with `model_comparison.py`, `model_comparison_wind.py` and `model_comparison_current.py`.

## Frames

| Frame | x | y | Rotation |
|---|---|---|---|
| World (earth-fixed) | East | North | counter-clockwise positive |
| Body (ship-fixed) | forward (bow) | **port** | counter-clockwise positive (z up) |

## Ship state (`fw.ships.vessel_state.VesselState`)

| Quantity | Unit | Frame | Positive | Notes |
|---|---|---|---|---|
| `x`, `y` | m | world | East, North | |
| `heading` | rad | world | counter-clockwise from East | 0 = East, π/2 = North; wrapped to [-π, π) |
| `u` (surge) | m/s | body, over ground | forward | |
| `v` (sway) | m/s | body, over ground | **to port** | `schema.SWAY_CONVENTION = "port_positive"` |
| `r` (yaw rate) | rad/s | — | counter-clockwise (bow to port) | |

## Actuators

| Quantity | Unit | Positive | Notes |
|---|---|---|---|
| Rudder command (`rudder_angle` in `DynamicsBase.step`) | rad | **the ship turns to port**, every ship and model | For the Tripoli's Z-drive the azimuth is physically to starboard. Maximum per ship: `ShipSpecifications.max_rudder_angle`, set in `ship_factory` (Myzako 75°, Tripoli 35°). |
| `rudder_action` (dataset, surrogate input) | [-1, 1] | the ship turns to port | Degrees = `rudder_action × rudder_max_deg`. `rudder_max_deg` is stored in the dataset and checkpoint. |
| Policy action `[rudder, thrust]` | [-1, 1] | rudder: the ship turns to port | `ActionMapper` maps thrust to `[min_thrust, max_thrust]` (Tripoli/Myzako: [0.1, 1.0]) |
| Thrust passed to `DynamicsBase.step` | normalized engine lever | ahead | Sent to FH's `setEngineLeverValue` unchanged |
| `thrust_action` (dataset, surrogate input) | [-1, 1] | ahead | The generators sample the full range, astern included |

**Toy models.** `Fossen` and `Nomoto` follow the turning direction (a positive rudder turns the ship to port). Their sway during a turn does **not** match real ship behaviour: at least at the start of the turn it goes toward the inside instead of the outside. The effect is small (drift under 2°), and for Fossen it is part of how the model turns, so flipping one coefficient would reverse the turn. Use UGent and FH as the references for sway. See the class docstrings.

## Environment

| Quantity | Unit | Frame | Positive / meaning |
|---|---|---|---|
| UKC | m | — | water under the keel |
| `wind_speed`, `wind_direction` (dataset) | m/s, rad | relative to the bow | **Apparent** wind of the recording ship, as FH's `WindState`. The direction is where the wind comes **from**, positive = **from starboard**. |
| `current_speed`, `current_direction` (dataset) | m/s, rad | world | Earth-fixed current. The direction is where it flows **to**, 0 = East, counter-clockwise. |
| Rays `ray_0` … `ray_345` | [0, 1] | body | `ray_k` points k° **clockwise** from the bow: `ray_90` = starboard beam, `ray_270` = port beam. 1 = boundary at the ship, 0 = nothing within range. Layout tag: `RAY_LAYOUT_VERSION = "bow0_clockwise_v2"`. World angle of a ray = heading − k. |

**Apparent wind.** In (ahead, starboard) components of the "comes from" vector: apparent = true + (u, −v). The −v appears because sway is positive to port, so a ship moving to port feels wind from port. Implemented in `uantwerp_surrogate.apparent_wind_for_state`, and in the wind test's `_true_wind_world` / `_apparent_wind_for`.

**Model encoding** (`env_encoding = "body_components"`, the default):

| Feature | Formula |
|---|---|
| `wind_from_ahead` | `ws·cos(wd)` |
| `wind_from_stbd` | `ws·sin(wd)` |
| `current_to_ahead` | `cs·cos(cd − heading)` |
| `current_to_stbd` | `−cs·sin(cd − heading)` |

## Surrogate data

| Item | Value |
|---|---|
| Time step | `SURROGATE_DT = 0.1` s |
| Targets | `delta_surge`, `delta_sway` (port positive), `delta_yaw_rate`, each over `SURROGATE_DT` in the body frame |
| Raw row layouts | `open_water` (8), `confined_ukc` (33), `full_env` (37). The definitions in `fw.surrogate.schema` are authoritative. |
| Identity in every `.npz` and checkpoint | `scenario`, `ray_layout`, `ship_model`, `rudder_max_deg`, `sway_convention`. Checked on load, so a mismatch is refused. |

Lookup CSVs carry **no** identity. Regenerate them together with the datasets.

**Compatibility policy.** There is no migration: data or checkpoints with a different identity are refused and must be regenerated or retrained. The generators' `_CHECKPOINT_VERSION` keeps an interrupted run from being resumed on incompatible chunks. A matching identity does not prove the data came from the same FH build or generator settings.

## FH (FH Simulator) API

| FH call | FH convention | Framework |
|---|---|---|
| `getShipPosition()` | world x, y [m] | as is |
| `getShipHeading()` | degrees, counter-clockwise from East (`FH_HEADING_CONVENTION = "math"`) | `radians(...)`, wrapped |
| `getShipVelocityOverGround()` | **body** frame despite the name; `.y` is port positive | `u = .x`, `v = .y` (no sign change) |
| `getShipYawRate()` | deg/s, counter-clockwise positive | `radians(...)` |
| `setInitialSwayVelocity(v)` | port positive | `+v` |
| Azipod `setDesiredDirection(deg)` (Tripoli) | + turns to **port** | `fh_rudder_sign = +1` |
| Rudder `setControlValue(val)` (Myzako), [-1, 1] | + turns to **starboard** | `fh_rudder_sign = −1` |
| `setInitialRudderAngle(deg)` | Myzako: same sign as `setControlValue` (measured 2026-10-19). Tripoli: no measurable effect on the azipods (measured 2026-10-19), so the azipods start at the exercise default. | `fh_rudder_sign(ship) × deg` |
| Rudder `getMaximumAngle()` (Myzako 37) | 75° = the generators' `RUDDER_MAX_DEG` and `ship_factory`'s Myzako maximum (measured 2026-10-19) | |
| `getWindState()` | Speed [m/s], Direction [deg]: apparent, coming from, + from starboard | `radians(Direction)` |

## Conversion points (the only places a sign or frame may change)

| Where | What changes |
|---|---|
| `fw.dynamics.fh_tabular_model.fh_rudder_sign(ship_model)` | Framework rudder sign → FH command sign. Used by `FhTabularModel`, `FhSimWrapper`, `FhSession` and the Myzako generators. |
| `fw.dynamics.ugent_surrogate` | UGent report frame (y to starboard) ↔ framework: `v` and `r` change sign at the boundary. |
| `fw.dynamics.uantwerp_surrogate.apparent_wind_for_state` | Recording ship's apparent wind → this ship's apparent wind. |
| `fw.surrogate.schema.select_and_encode` | Wind and current angles → ship-frame components. |
| `fw.environments.ray_features.ray_world_angles` | Clockwise ray bearing → counter-clockwise world angle. |
| `fh_heading_to_env_rad` / `env_rad_to_fh_heading_deg` (generators) | FH heading degrees ↔ framework radians. |
| `FhSimBridge` (`setInitialHeading(degrees(heading))`), `ShipSimulator`, `FhSession`, `FhSimWrapper` (`radians(getShipHeading())`) | The same heading conversion, written out directly. All of them assume `FH_HEADING_CONVENTION = "math"`. |

## Adding a ship

1. Measure FH's rudder sign with a short dataset and `check_rudder_sign.py` (add `--negated` if the generator flips the sign). Then add the ship to `_FH_RUDDER_SIGNS` in `fh_tabular_model.py`; `fh_rudder_sign` refuses ships that are not listed.
2. Check that FH's sway is still port positive with `check_sway_sign.py` (expected slope about −1 after the 2026-10-19 fix).
3. Check the sign of the initial rudder angle with `check_initial_rudder_sign.py --ship <SHIP>`.
4. Run `python -m fw.surrogate.validation.check_model_conventions --ship <SHIP>` (from `src`): every model, FH included, must turn the same way for the same command.
5. If the ship has a new kind of actuator, add a case to `tests/surrogate/test_conventions.py`.
6. Set the ship's maximum rudder angle in `ship_factory` (every ship branch must set it explicitly).
