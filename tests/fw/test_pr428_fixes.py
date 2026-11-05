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


def _execute(monkeypatch, blocks, ua_preset):
    """
    PipelineScheduler.execute() with config.py's UAntwerp preset set to
    `ua_preset` and a stubbed _run_block that resolves the block's config and
    raises like the real one for an unknown preset; returns the metrics.
    """

    from fw.configuration import config as config_module
    from fw.scheduling import pipeline_env
    from fw.scheduling.pipeline_scheduler import PipelineScheduler
    from fw.surrogate.presets import get_preset

    def load():
        """
        config.py with the given UAntwerp preset and dynamics model.
        """

        cfg = config_module.load_config_parameters()
        cfg["ua_surrogate_config"] = {"preset": ua_preset}
        return cfg

    monkeypatch.setattr(pipeline_env, "load_config_parameters", load)

    def run_block(self, block, *_args):
        """
        Stand-in for _run_block: fails, as building the env would, on an unknown preset.
        """

        cfg = pipeline_env.block_framework_config(block, pipeline_env.resolve_block_profile(block))
        if cfg["dynamics_model"].name == "UANTWERP" and block.env_type == "PySimEnv":
            get_preset(cfg["ua_surrogate_config"]["preset"])
        return {"duration": 0.0}

    monkeypatch.setattr(PipelineScheduler, "_run_block", run_block)
    return PipelineScheduler(agent_mgr=None, blocks=blocks).execute()


def _blocks(*raw):
    """
    Parsed pipeline blocks (names b0, b1, ...).
    """

    from fw.scheduling.run_loader import _parse_spec

    return _parse_spec({"blocks": [dict(name=f"b{i}", action="train", **r) for i, r in enumerate(raw)]},
                       "test").blocks


def test_execute_runs_a_lunar_block_under_a_full_preset_config(monkeypatch):
    """
    Pre-flight only checks PySimEnv-based blocks: a Lunar block next to a UA full block runs.
    """

    blocks = _blocks(dict(env_type="PySimEnv", dynamics_model="UANTWERP", ship_model="MYZAKO_37",
                          wind=True, current=True),
                     dict(env_type="LunarLander", dynamics_model="UANTWERP"))
    metrics = _execute(monkeypatch, blocks, "myzako_full")
    assert [metrics[b]["status"] for b in ("b0", "b1")] == ["success", "success"]


def test_execute_leaves_a_preset_typo_to_its_block(monkeypatch):
    """
    An unknown preset is not raised by pre-flight: that block fails, the next one still runs.
    """

    blocks = _blocks(dict(env_type="PySimEnv", dynamics_model="UANTWERP", ship_model="MYZAKO_37"),
                     dict(env_type="PySimEnv"))
    metrics = _execute(monkeypatch, blocks, "myzako_confinedd")
    assert metrics["b0"]["status"] == "failure" and "myzako_confinedd" in metrics["b0"]["error"]
    assert metrics["b1"]["status"] == "success"


def test_execute_stops_before_any_block_on_a_wind_current_mismatch(monkeypatch):
    """
    A full preset with the flags off in a later block stops the run before the first block.
    """

    blocks = _blocks(dict(env_type="PySimEnv"),
                     dict(env_type="PySimEnv", dynamics_model="UANTWERP", ship_model="MYZAKO_37"))
    with pytest.raises(ValueError, match="must be enabled"):
        _execute(monkeypatch, blocks, "myzako_full")
