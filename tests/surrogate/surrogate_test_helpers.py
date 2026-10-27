"""
Helpers shared by the surrogate contract tests (synthetic datasets, tiny
models, vessel states). Imported by the tests via conftest.py, which puts this
folder on sys.path.
"""

from __future__ import annotations

import dataclasses

import numpy as np

from fw.surrogate.schema import DatasetIdentity, SWAY_CONVENTION, get_scenario, npz_metadata
from fw.environments.ray_features import RAY_LAYOUT_VERSION


def make_state(**values):
    """
    A fw.ships.vessel_state.VesselState from keyword values; fields not given
    (e.g. `terminated` / `truncated` flags) are set to False / 0.0.
    """

    from fw.ships.vessel_state import VesselState

    names = {f.name for f in dataclasses.fields(VesselState)}
    unknown = sorted(set(values) - names)
    if unknown:
        raise TypeError(f"VesselState has no field(s) {unknown}; fields: {sorted(names)}")
    kwargs = {}
    for f in dataclasses.fields(VesselState):
        if f.name in values:
            kwargs[f.name] = values[f.name]
        elif f.default is not dataclasses.MISSING or f.default_factory is not dataclasses.MISSING:
            continue
        else:
            kwargs[f.name] = False if f.name in ("terminated", "truncated") else 0.0
    return VesselState(**kwargs)


def identity(scenario: str, ship_model: str = "MYZAKO_37", rudder_max_deg: float = 75.0) -> DatasetIdentity:
    """A valid dataset identity for synthetic data."""

    return DatasetIdentity(scenario=scenario, ray_layout=RAY_LAYOUT_VERSION, ship_model=ship_model,
                           rudder_max_deg=float(rudder_max_deg), sway_convention=SWAY_CONVENTION)


def synthetic_rows(scenario_name: str, n_episodes: int, steps: int, seed: int):
    """
    Random but plausible raw rows of `scenario_name` in whole episodes, with
    targets that depend on the inputs (so a model can learn something).

    Returns:
        (X float32 (N, raw width), Y float32 (N, 3), episode_id int64 (N,)).
    """

    sc = get_scenario(scenario_name)
    rng = np.random.default_rng(seed)
    xs, ys, eps = [], [], []
    for ep in range(n_episodes):
        x = np.zeros((steps, sc.raw_input_dim), dtype=np.float64)
        x[:, sc.col("x")] = rng.uniform(0.0, 500.0) + np.arange(steps) * 0.3
        x[:, sc.col("y")] = rng.uniform(0.0, 200.0)
        x[:, sc.col("heading")] = rng.uniform(-np.pi, np.pi)
        x[:, sc.col("surge")] = rng.uniform(0.5, 5.0)
        x[:, sc.col("sway")] = rng.normal(0.0, 0.1)
        x[:, sc.col("yaw_rate")] = rng.normal(0.0, 0.01)
        x[:, sc.col("rudder_action")] = rng.uniform(-1.0, 1.0)
        x[:, sc.col("thrust_action")] = rng.uniform(0.0, 1.0)
        if sc.includes_ukc:
            x[:, sc.col("ukc")] = rng.uniform(2.0, 20.0)
        if sc.includes_wind_current:
            x[:, sc.col("wind_speed")] = rng.uniform(0.0, 10.0)
            x[:, sc.col("wind_direction")] = rng.uniform(-np.pi, np.pi)
            x[:, sc.col("current_speed")] = rng.uniform(0.0, 1.0)
            x[:, sc.col("current_direction")] = rng.uniform(-np.pi, np.pi)
        if sc.includes_rays:
            for c in sc.raw_columns:
                if c.startswith("ray_"):
                    x[:, sc.col(c)] = rng.uniform(0.0, 0.5)
        y = np.column_stack([
            0.01 * x[:, sc.col("thrust_action")] - 0.002 * x[:, sc.col("surge")],
            -0.003 * x[:, sc.col("rudder_action")] + rng.normal(0.0, 1e-4, steps),
            0.0005 * x[:, sc.col("rudder_action")] * x[:, sc.col("surge")],
        ])
        xs.append(x)
        ys.append(y)
        eps.append(np.full(steps, ep, dtype=np.int64))
    return (np.concatenate(xs).astype(np.float32), np.concatenate(ys).astype(np.float32),
            np.concatenate(eps))


def write_splits(folder, scenario_name: str, ship_model: str = "MYZAKO_37", rudder_max_deg: float = 75.0,
                 seed: int = 0, provenance=None):
    """Write train/val/test .npz files (disjoint episodes) of synthetic data; returns their paths."""

    ident = identity(scenario_name, ship_model, rudder_max_deg)
    paths = []
    for i, name in enumerate(("train", "val", "test")):
        x, y, ep = synthetic_rows(scenario_name, n_episodes=6 if name == "train" else 2, steps=40,
                                  seed=seed * 10 + i)
        path = folder / f"{scenario_name}_{name}.npz"
        extra = {}
        if provenance is not None:
            from fw.surrogate.schema import provenance_metadata
            extra = provenance_metadata(provenance)
        np.savez(path, X=x, Y=y, episode_id=ep + 100 * i, **npz_metadata(ident), **extra)
        paths.append(path)
    return paths


def train_tiny(folder, scenario_name: str, extra_args=(), **split_kwargs):
    """Train a one-epoch model on synthetic splits; returns (checkpoint path, split paths)."""

    from fw.surrogate import train_surrogate

    train, val, test = write_splits(folder, scenario_name, **split_kwargs)
    ckpt = folder / f"{scenario_name}_model.pth"
    train_surrogate.main(["--train-file", str(train), "--val-file", str(val), "--test-file", str(test),
                          "--epochs", "1", "--patience", "2", "--batch-size", "64", "--device", "cpu",
                          "--checkpoint", str(ckpt), *extra_args])
    return ckpt, (train, val, test)
