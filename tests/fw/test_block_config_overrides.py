"""
Block-level config.py overrides in pipeline specs (scenario, surrogate
configs), load-time validation of pipeline specs, and the reward-evaluation
environment config.
"""

import json
import pytest
import logging

from pathlib import Path
from fw.scheduling import pipeline_env
from fw.scheduling.pipeline_scheduler import PipelineScheduler
from fw.scheduling.run_loader import PipelineSpecError, load_pipeline_spec
from fw.configuration.env_factory import fh_sim_env_name, multi_sim_env_name, py_sim_env_name

SPEC_DIR = Path(pipeline_env.__file__).resolve().parents[1]

_UA_BLOCK = {"name": "b0", "env_type": "PySimEnv", "ship_model": "MYZAKO_37", "dynamics_model": "UANTWERP",
             "profile": "uantwerp_surrogate_profile", "wind": True, "current": True}


def _write_spec(tmp_path, block=None, top=None, **block_keys):
    """
    Write a one-block spec (default: a UAntwerp PySimEnv block) with the given
    extra block keys and top-level keys; returns its path.
    """

    block = {**(block or _UA_BLOCK), **block_keys}
    path = tmp_path / "spec.json"
    path.write_text(json.dumps({"blocks": [block], **(top or {})}), encoding="utf-8")
    return str(path)


def _resolved(spec_path, index=0):
    """
    The resolved framework config of block `index` of a spec file.
    """

    block = load_pipeline_spec(spec_path).blocks[index]
    return pipeline_env.block_framework_config(block, pipeline_env.resolve_block_profile(block))


@pytest.fixture
def config_py_ua(monkeypatch):
    """
    Set config.py's ua_surrogate_config (as seen by the pipeline) to a given dict.
    """

    from fw.configuration import config as config_module

    def set_config(ua_config):
        """
        Make load_config_parameters return `ua_config` as ua_surrogate_config.
        """

        def load():
            """
            config.py with the patched ua_surrogate_config.
            """

            cfg = config_module.load_config_parameters()
            cfg["ua_surrogate_config"] = dict(ua_config)
            return cfg

        monkeypatch.setattr(pipeline_env, "load_config_parameters", load)

    return set_config


@pytest.mark.parametrize("spec_file", sorted(p.name for p in SPEC_DIR.glob("*.json")))
def test_every_shipped_spec_loads(spec_file, caplog):
    """
    All pipeline JSON files in src/fw pass the load-time checks, without a
    block key that has no effect (e.g. a scenario on a PySimEnv block).
    """

    raw = json.loads((SPEC_DIR / spec_file).read_text(encoding="utf-8"))
    if "blocks" not in raw:
        pytest.skip("not a pipeline spec")
    with caplog.at_level(logging.WARNING, logger="fw.scheduling.run_loader"):
        load_pipeline_spec(str(SPEC_DIR / spec_file))
    assert "has no effect" not in caplog.text


def test_ua_full_spec_resolves_myzako_full():
    """
    The reported bug: ua_full_fh_pipeline.json's first block uses myzako_full.
    """

    cfg = _resolved(str(SPEC_DIR / "ua_full_fh_pipeline.json"))
    assert cfg["ua_surrogate_config"]["preset"] == "myzako_full"
    assert cfg["ua_surrogate_config"]["wind_current"] == "random"


def test_override_merges_over_config_py(tmp_path, config_py_ua):
    """
    Keys the block does not set keep config.py's value.
    """

    config_py_ua({"preset": "myzako_confined", "env_smoothing_tau": 5.0, "wind_current": "random"})
    cfg = _resolved(_write_spec(tmp_path, ua_surrogate_config={"preset": "myzako_full"}))
    assert cfg["ua_surrogate_config"] == {"preset": "myzako_full", "env_smoothing_tau": 5.0, "wind_current": "random"}


def test_other_preset_drops_preset_specific_keys(tmp_path, config_py_ua):
    """
    config.py's checkpoint/lookup_csv are not inherited by another preset.
    """

    config_py_ua({"preset": "myzako_confined", "checkpoint": "c.pth", "lookup_csv": "l.csv.gz",
                  "wind_current": "random"})
    cfg = _resolved(_write_spec(tmp_path, ua_surrogate_config={"preset": "myzako_full"}))
    assert cfg["ua_surrogate_config"] == {"preset": "myzako_full", "wind_current": "random"}


def test_same_preset_keeps_preset_specific_keys(tmp_path, config_py_ua):
    """
    Without a preset change, config.py's checkpoint/lookup_csv stay.
    """

    config_py_ua({"preset": "myzako_full", "checkpoint": "c.pth", "lookup_csv": "l.csv.gz"})
    cfg = _resolved(_write_spec(tmp_path, ua_surrogate_config={"wind_current": "lookup"}))
    assert cfg["ua_surrogate_config"] == {"preset": "myzako_full", "checkpoint": "c.pth", "lookup_csv": "l.csv.gz",
                                          "wind_current": "lookup"}


def test_empty_override_keeps_config_py(tmp_path):
    """
    {} and an absent key both mean config.py's ua_surrogate_config.
    """

    from fw.configuration.config import load_config_parameters

    expected = load_config_parameters()["ua_surrogate_config"]
    assert _resolved(_write_spec(tmp_path, ua_surrogate_config={}))["ua_surrogate_config"] == expected
    assert _resolved(_write_spec(tmp_path))["ua_surrogate_config"] == expected


def test_override_reaches_env_config_builder(tmp_path):
    """
    The merged config is what the PySimEnv config builder passes on.
    """

    cfg = _resolved(_write_spec(tmp_path, ua_surrogate_config={"preset": "myzako_full"}))
    env_cfg = pipeline_env._CONFIG_BUILDERS[py_sim_env_name](cfg)
    assert env_cfg["ua_surrogate_config"]["preset"] == "myzako_full"


def test_ugent_override_merges(tmp_path):
    """
    ugent_surrogate_config is merged over config.py's like the UA config.
    """

    block = {"name": "b0", "env_type": "PySimEnv", "ship_model": "TRIPOLI", "dynamics_model": "UGENT_SURROGATE",
             "profile": "ugent_surrogate_profile"}
    cfg = _resolved(_write_spec(tmp_path, block=block, ugent_surrogate_config={"ukc_percent": 200}))
    assert cfg["ugent_surrogate_config"]["ukc_percent"] == 200


def test_scenario_override_reaches_fh_config(tmp_path):
    """
    A block scenario reaches the FhSimEnv config.
    """

    block = {"name": "b0", "env_type": "FhSimEnv", "ship_model": "MYZAKO_37", "dynamics_model": "FH_TABULAR",
             "profile": "fh_simulator_profile"}
    cfg = _resolved(_write_spec(tmp_path, block=block, scenario="my_scenario"))
    assert pipeline_env._CONFIG_BUILDERS[fh_sim_env_name](cfg)["scenario"] == "my_scenario"


def test_scenario_on_pysim_block_warns(tmp_path, caplog):
    """
    PySimEnv does not read scenario: loading warns.
    """

    with caplog.at_level(logging.WARNING, logger="fw.scheduling.run_loader"):
        load_pipeline_spec(_write_spec(tmp_path, scenario="DDShip_scenario1_nowindnocurrent"))
    assert "has no effect" in caplog.text


def test_resolved_block_configs_records_preset(tmp_path):
    """
    The run record lists each block's effective preset.
    """

    spec = load_pipeline_spec(_write_spec(tmp_path, ua_surrogate_config={"preset": "myzako_full"}))
    assert pipeline_env.resolved_block_configs(spec.blocks)["b0"]["ua_surrogate_config"]["preset"] == "myzako_full"


@pytest.mark.parametrize("block_keys, match", [
    ({"ua_surrogate_config": {"preset": "myzako_fulll"}}, "Unknown UAntwerp surrogate preset"),
    ({"ua_surrogate_config": {"preset": "myzako_full", "bogus": 1}}, r"unknown key\(s\) \['bogus'\]"),
    ({"ua_surrogate_config": "myzako_full"}, "must be an object"),
    ({"ua_surrogate_config": {"preset": "tripoli_full"}}, "is for TRIPOLI"),
    ({"ugent_surrogate_config": {"ukc_percent": 363}}, "only read by dynamics model UGENT_SURROGATE"),
    ({"scenario": 3}, "'scenario' must be a non-empty string"),
    ({"ua_surogate_config": {"preset": "myzako_full"}}, r"unknown key\(s\) \['ua_surogate_config'\]"),
    ({"transfer": {"from": "x"}}, r"unknown key\(s\) \['from'\] in transfer"),
    ({"stop_condition": {"max_steps": 5}}, r"unknown key\(s\) \['max_steps'\] in stop_condition"),
    ({"transfer": "x"}, "'transfer' must be an object"),
    ({"transfer": []}, "'transfer' must be an object, got list"),
    ({"stop_condition": False}, "'stop_condition' must be an object, got bool"),
    ({"profile_overrides": "x"}, "'profile_overrides' must be an object"),
    ({"ship_model": "TRIPOLI"}, "preset 'myzako_confined' is for MYZAKO_37, but this block uses TRIPOLI"),
    ({"action": "evaluate_reward", "reward_configs": [{"type": "PySimEnv", "wind": True}]},
     r"unknown key\(s\) \['wind'\] in the reward config"),
    ({"action": "evaluate_reward", "reward_configs": [{"type": "FhSimEnv"}]}, "they must match"),
    ({"action": "evaluate_reward", "reward_configs": [{"reward_name": "r1"}]}, "'type' is required"),
    ({"action": "evaluate_reward", "reward_configs": ["r1"]}, r"reward_configs\[0\]: must be an object"),
    ({"action": "evaluate_reward", "reward_configs": {"type": "PySimEnv"}}, "'reward_configs' must be a list"),
    ({"action": "evaluate_reward", "reward_configs": [{"type": "PySimEnv", "reward_weights": 1}]},
     "'reward_weights' must be an object"),
])
def test_invalid_block_fails_at_load(tmp_path, block_keys, match):
    """
    Invalid block contents are rejected when the spec is loaded.
    """

    with pytest.raises(PipelineSpecError, match=match):
        load_pipeline_spec(_write_spec(tmp_path, **block_keys))


def test_ua_config_on_non_uantwerp_block_fails(tmp_path):
    """
    ua_surrogate_config on a FOSSEN block is an error, not silently ignored.
    """

    with pytest.raises(PipelineSpecError, match="only read by dynamics model UANTWERP"):
        load_pipeline_spec(_write_spec(tmp_path, dynamics_model="FOSSEN",
                                       ua_surrogate_config={"preset": "myzako_full"}))


def test_non_dict_block_and_unknown_top_level_key_fail(tmp_path):
    """
    A block that is not an object, and an unknown top-level key, are rejected.
    """

    path = tmp_path / "spec.json"
    path.write_text(json.dumps({"blocks": ["b0"]}), encoding="utf-8")
    with pytest.raises(PipelineSpecError, match="a block must be a JSON object"):
        load_pipeline_spec(str(path))
    with pytest.raises(PipelineSpecError, match=r"unknown key\(s\) \['on_failure'\] in the specification"):
        load_pipeline_spec(_write_spec(tmp_path, top={"on_failure": "abort"}))


def test_eval_config_uses_framework_config():
    """
    A reward-evaluation env gets the framework config plus the reward settings,
    not the bare reward entry.
    """

    from fw.scheduling.trainer import Trainer
    from fw.configuration.config import load_config_parameters

    framework_config = load_config_parameters()
    framework_config["ua_surrogate_config"] = {"preset": "myzako_full"}
    cfg = Trainer._build_eval_config(py_sim_env_name, framework_config, "r1", {"w": 1.0})
    assert cfg["start_pos"] == framework_config["start_pos"]
    assert cfg["ua_surrogate_config"] == {"preset": "myzako_full"}
    assert (cfg["reward_name"], cfg["reward_weights"]) == ("r1", {"w": 1.0})


def test_eval_inner_trainer_gets_config(monkeypatch):
    """
    The per-reward Trainer is built with the outer Trainer's config, not config.py.
    """

    from fw.scheduling import trainer as trainer_module

    seen = []

    class FakeTrainer(trainer_module.Trainer):
        """
        Records the config of each Trainer built inside evaluate_reward.
        """

        def __init__(self, agent_type, envs, profile, scheduler_type, config=None):
            """
            Record `config` and stop the evaluation of this reward config.
            """

            seen.append(config)
            raise RuntimeError("stop")

    real_trainer = trainer_module.Trainer
    outer = real_trainer.__new__(real_trainer)
    outer._agent_type, outer._profile, outer._config = "PPO", None, {"marker": True}
    monkeypatch.setattr(real_trainer, "_build_eval_config", staticmethod(lambda *a: {}))
    monkeypatch.setattr(real_trainer, "_build_eval_environment", staticmethod(lambda *a: object()))
    monkeypatch.setattr(trainer_module, "Trainer", FakeTrainer)
    results = real_trainer.evaluate_reward(outer, [{"type": "PySimEnv"}], verbose=False)
    assert seen == [{"marker": True}]
    assert results[0]["status"] == "failed: stop"


def test_uantwerp_on_colav_fails(tmp_path):
    """
    Model.UANTWERP is not supported on the COLAV environments.
    """

    with pytest.raises(PipelineSpecError, match="UANTWERP is not supported on PySimColavEnv"):
        load_pipeline_spec(_write_spec(tmp_path, env_type="PySimColavEnv"))


def test_comment_keys_are_allowed(tmp_path):
    """
    Keys starting with "_" are comments, at every level; they are not passed to
    the surrogate.
    """

    spec = load_pipeline_spec(_write_spec(
        tmp_path, top={"_comment": "x"}, _note="x", stop_condition={"_c": 1},
        ua_surrogate_config={"preset": "myzako_full", "_why": "full env"},
        action="evaluate_reward", reward_configs=[{"type": "PySimEnv", "_c": 1}]))
    assert spec.blocks[0].config_overrides["ua_surrogate_config"] == {"preset": "myzako_full"}


def test_multisim_override_reaches_env_config_builder(tmp_path):
    """
    The override also reaches the MultiSimEnv config.
    """

    cfg = _resolved(_write_spec(tmp_path, env_type="MultiSimEnv", ua_surrogate_config={"preset": "myzako_full"}))
    assert pipeline_env._CONFIG_BUILDERS[multi_sim_env_name](cfg)["ua_surrogate_config"]["preset"] == "myzako_full"


def test_ugent_b01_override_merges(tmp_path):
    """
    ugent_b01_surrogate_config is merged over config.py's like the others.
    """

    block = {"name": "b0", "env_type": "PySimEnv", "profile": "ugent_b01_surrogate_profile"}
    cfg = _resolved(_write_spec(tmp_path, block=block, ugent_b01_surrogate_config={"ukc_percent": 200}))
    assert cfg["ugent_b01_surrogate_config"]["ukc_percent"] == 200


def test_resolved_block_configs_is_json_serializable(tmp_path):
    """
    The run record (spec plus resolved_block_config, as main_pipeline builds it) is JSON.
    """

    import dataclasses

    spec = load_pipeline_spec(_write_spec(tmp_path, ua_surrogate_config={"preset": "myzako_full"}))
    snapshot = dataclasses.asdict(spec)
    snapshot["resolved_block_config"] = pipeline_env.resolved_block_configs(spec.blocks)
    assert json.loads(json.dumps(snapshot))["resolved_block_config"]["b0"]["ua_surrogate_config"]["preset"] == \
        "myzako_full"


def test_preflight_rejects_full_override_without_wind_current(tmp_path):
    """
    A block's myzako_full override with wind/current off fails before any block runs.
    """

    spec = load_pipeline_spec(_write_spec(tmp_path, wind=False, current=False,
                                          ua_surrogate_config={"preset": "myzako_full"}))
    with pytest.raises(ValueError, match="must be enabled"):
        PipelineScheduler._preflight(spec.blocks)


def test_preflight_rejects_preset_ship_mismatch_without_override():
    """
    Blocks built in Python get the preset/ship check in _preflight, also
    without a ua_surrogate_config override (config.py's preset).
    """

    from fw.scheduling.run_spec import PipelineBlock

    block = PipelineBlock(name="b0", env_type="PySimEnv", profile="uantwerp_surrogate_profile",
                          ship_model="TRIPOLI", dynamics_model="UANTWERP")
    with pytest.raises(ValueError, match="but this block uses TRIPOLI"):
        PipelineScheduler._preflight([block])


def test_run_evaluate_reward_uses_block_override(tmp_path, monkeypatch):
    """
    End to end through PipelineScheduler._run_evaluate_reward: the reward
    evaluation env gets the block's merged ua_surrogate_config.
    """

    from types import SimpleNamespace
    from fw.scheduling import trainer as trainer_module

    spec = load_pipeline_spec(_write_spec(
        tmp_path, action="evaluate_reward", ua_surrogate_config={"preset": "myzako_full"},
        reward_configs=[{"type": "PySimEnv", "reward_name": "r1", "reward_weights": {"forward": -1.0}}]))
    block = spec.blocks[0]
    profile = pipeline_env.resolve_block_profile(block)
    framework_config = pipeline_env.block_framework_config(block, profile)

    def init(self, agent_type, envs, profile_, scheduler_type, config=None):
        """
        Minimal Trainer: no agent manager or scheduler.
        """

        self._agent_type, self._envs, self._profile, self._config = agent_type, envs, profile_, config

    seen = []

    def build_env(canonical_name, config):
        """
        Record the eval env config, then stop this reward config.
        """

        seen.append((canonical_name, config))
        raise RuntimeError("stop")

    monkeypatch.setattr(trainer_module.Trainer, "__init__", init)
    monkeypatch.setattr(trainer_module.Trainer, "_build_eval_environment", staticmethod(build_env))
    monkeypatch.setattr(trainer_module.Trainer, "export_evaluation_results", staticmethod(lambda *a: None))
    scheduler = SimpleNamespace(agent_mgr=SimpleNamespace(agent_type="PPO"),
                                _build_stop_condition=PipelineScheduler._build_stop_condition)

    PipelineScheduler._run_evaluate_reward(scheduler, block, object(), profile, framework_config)
    [(canonical_name, env_config)] = seen
    assert canonical_name == py_sim_env_name
    assert env_config["ua_surrogate_config"]["preset"] == "myzako_full"
    assert env_config["ua_surrogate_config"]["wind_current"] == framework_config["ua_surrogate_config"]["wind_current"]
    assert (env_config["reward_name"], env_config["reward_weights"]) == ("r1", {"forward": -1.0})


def test_evaluate_reward_rejects_unknown_reward_key():
    """
    Trainer.evaluate_reward (also used by main.py) rejects unknown keys
    before any training starts.
    """

    from fw.scheduling.trainer import Trainer

    trainer = Trainer.__new__(Trainer)
    with pytest.raises(ValueError, match=r"unknown key\(s\) \['wind'\]"):
        trainer.evaluate_reward([{"type": "PySimEnv", "wind": True}], verbose=False)


def test_config_py_preset_typo_is_left_to_the_block(tmp_path, config_py_ua):
    """
    A preset typo in config.py does not stop the spec from loading (the block
    fails when it runs); a typo in the block's own preset does.
    """

    config_py_ua({"preset": "myzako_confinedd"})
    load_pipeline_spec(_write_spec(tmp_path))
    with pytest.raises(PipelineSpecError, match="Unknown UAntwerp surrogate preset 'myzako_fulll'"):
        load_pipeline_spec(_write_spec(tmp_path, ua_surrogate_config={"preset": "myzako_fulll"}))
