"""
Regression tests for review findings on the wind/current pull request:
importability of the FH validation scripts, early pipeline validation of the
wind/current flags, and the dataset-layout check of the scale tool.
"""

import importlib

import numpy as np
import pytest


@pytest.mark.parametrize("module", [
    "fw.surrogate.validation.model_comparison",
    "fw.surrogate.validation.model_comparison_wind",
    "fw.surrogate.validation.model_comparison_current",
    "fw.surrogate.sim_fh_dataset_generator",
])
def test_fh_validation_scripts_import(module):
    """
    The FH validation scripts import without FH (they broke when a re-export was removed).
    """

    importlib.import_module(module)


def _block(**extra):
    """
    Minimal valid pipeline block, PySimEnv training with Fossen.
    """

    block = {"name": "b1", "env_type": "PySimEnv", "action": "train"}
    block.update(extra)
    return block


@pytest.mark.parametrize("wind, current", [(True, False), (False, True)])
def test_pipeline_rejects_unequal_wind_and_current_at_load(wind, current):
    """
    A block with wind != current fails when the spec is loaded, not when the env is built.
    """

    from fw.scheduling.run_loader import PipelineSpecError, _parse_spec

    with pytest.raises(PipelineSpecError, match="together"):
        _parse_spec({"blocks": [_block(wind=wind, current=current)]}, "test")


def test_pipeline_accepts_equal_wind_and_current():
    """
    Both on (or both off) is accepted and carried into the block.
    """

    from fw.scheduling.run_loader import _parse_spec

    spec = _parse_spec({"blocks": [_block(wind=True, current=True)]}, "test")
    assert spec.blocks[0].wind and spec.blocks[0].current


def test_scale_tool_refuses_a_non_full_env_csv(tmp_path):
    """
    A confined_ukc CSV (37 columns, wider than a full_env raw row) is refused.
    """

    from fw.surrogate.compute_wind_current_obs_scales import load_env_columns
    from fw.surrogate.schema import get_scenario

    width = get_scenario("confined_ukc").raw_input_dim + 3 + 1
    path = tmp_path / "confined.csv"
    np.savetxt(path, np.ones((3, width)), delimiter=",")
    with pytest.raises(ValueError, match="full_env"):
        load_env_columns(path)


def test_current_line_follower_stays_in_range():
    """
    The direction is always in [-pi, pi), also for a line along -x (where atan2 can give +pi).
    """

    from fw.dynamics.uantwerp_surrogate import CurrentLineFollower

    f = CurrentLineFollower(-1.0, 10.0)
    for _ in range(50):
        d = f.step(0.0, 1.0, 0.1)
        assert -np.pi <= d < np.pi



def _pipeline_blocks(**block):
    """
    Parsed pipeline blocks for one block with the given fields.
    """

    from fw.scheduling.run_loader import _parse_spec

    return _parse_spec({"blocks": [_block(**block)]}, "test").blocks


@pytest.fixture
def full_preset_config(monkeypatch):
    """
    Pipeline config whose UAntwerp surrogate uses the myzako_full preset.
    """

    from fw.configuration import config as config_module
    from fw.scheduling import pipeline_env

    def load():
        """
        config.py with the myzako_full preset.
        """

        cfg = config_module.load_config_parameters()
        cfg["ua_surrogate_config"] = {"preset": "myzako_full"}
        return cfg

    monkeypatch.setattr(pipeline_env, "load_config_parameters", load)


def test_preflight_rejects_full_preset_without_wind_current(full_preset_config):
    """
    A UAntwerp full_env block with the flags off fails before any block runs.
    """

    from fw.scheduling.pipeline_scheduler import PipelineScheduler

    blocks = _pipeline_blocks(dynamics_model="UANTWERP", ship_model="MYZAKO_37")
    with pytest.raises(ValueError, match="must be enabled"):
        PipelineScheduler._preflight(blocks)
    PipelineScheduler._preflight(_pipeline_blocks(dynamics_model="UANTWERP", ship_model="MYZAKO_37",
                                                  wind=True, current=True))


def test_preflight_rejects_simsync_with_wind_current():
    """
    simSync with the flags on fails before any block runs; without them it passes.
    """

    from fw.scheduling.pipeline_scheduler import PipelineScheduler

    with pytest.raises(ValueError, match="does not support wind and current"):
        PipelineScheduler._preflight(_pipeline_blocks(env_type="FhSimSyncEnv", wind=True, current=True))
    PipelineScheduler._preflight(_pipeline_blocks(env_type="FhSimSyncEnv"))


def test_preflight_leaves_other_errors_to_the_block(monkeypatch):
    """
    A block whose configuration cannot be resolved is not rejected up front (unchanged behaviour).
    """

    from fw.scheduling import pipeline_scheduler
    from fw.scheduling.pipeline_scheduler import PipelineScheduler

    def broken(_block):
        """
        Profile resolution that fails.
        """

        raise KeyError("no such profile")

    monkeypatch.setattr(pipeline_scheduler, "resolve_block_profile", broken)
    PipelineScheduler._preflight(_pipeline_blocks(wind=True, current=True))


def test_bridge_logs_the_replaced_scenario(tmp_path, caplog):
    """
    With wind_current the configured Myzako scenario is replaced by the wind/current base, and this is logged.
    """

    import logging
    from fw.ships.ship_factory import ShipModel
    from fw.simulators.fh_sim_bridge import FhSimBridge

    with caplog.at_level(logging.INFO, logger="fw.simulators.fh_sim_bridge"):
        FhSimBridge(ship_model=ShipModel.MYZAKO_37, output_path=str(tmp_path), wind_current=True).dispose()
        FhSimBridge(ship_model=ShipModel.MYZAKO_37, output_path=str(tmp_path), wind_current=False).dispose()
    lines = [r.getMessage() for r in caplog.records if "replaces the configured scenario" in r.getMessage()]
    assert len(lines) == 1 and "DDShip_scenario3_windcurrent" in lines[0]
