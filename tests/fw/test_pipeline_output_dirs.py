"""
Per-block output subfolders of the JSON pipeline: agent output paths,
AgentManager wiring, visualize folders and block-name validation.
"""

from __future__ import annotations

import os
import pytest
import numpy as np
import gymnasium as gym

from types import SimpleNamespace
from fw.scheduling.agent_manager import AgentManager
from fw.policies.single_ppo_agent import SinglePPOAgent
from fw.configuration.stop_condition import StopCondition
from fw.scheduling.pipeline_scheduler import PipelineScheduler
from fw.configuration.config import PPO_POLICY_NAME, PolicyStrategy
from fw.scheduling.run_spec import PipelineBlock, TransferSpec, block_output_subdir, validate_block_names


class _DummyEnv:
    """
    Minimal env stub for SinglePPOAgent.
    """

    type_name = "PySimEnv"
    training_name = "blk"
    observation_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
    action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)


class _NoIterationLog:
    """
    TrainingIterationLogger stub, so no CSV is written to src/logs.
    """

    path = "unused"

    def __init__(self, **_kwargs):
        """
        Accept the TrainingIterationLogger arguments.
        """

    def log_iteration(self, **_kwargs):
        """
        Ignore the row.
        """

    def save(self):
        """
        Write nothing.
        """


def _stub_learn(agent, monkeypatch, train=None):
    """
    Stub rollouts, PPO updates, policy saving and metric output of `agent`.

    Returns:
        (saved policy names, (dir, file name) of each metrics CSV dump).
    """

    import fw.policies.single_ppo_agent as single_ppo_agent_module

    episode = {"rewards": np.array([0.1, 0.2], dtype=np.float32),
               "advantages": np.array([0.0, 0.1], dtype=np.float32)}
    saved, dumped = [], []
    monkeypatch.setattr(agent, "run_parallel_episodes", lambda _n: [episode])
    monkeypatch.setattr(agent, "train", train or (lambda _data: 0.0))
    monkeypatch.setattr(agent, "save_policy", lambda name: saved.append(name))
    monkeypatch.setattr(single_ppo_agent_module, "dump_metrics_to_csv",
                        lambda model_dir, name, _metrics: dumped.append((model_dir, name)))
    monkeypatch.setattr(single_ppo_agent_module, "TrainingIterationLogger", _NoIterationLog)
    return saved, dumped


def test_output_paths_default_to_model_dir(tmp_path):
    """
    Without a subfolder, outputs go to model_dir.
    """

    agent = SinglePPOAgent(environment=_DummyEnv(), model_dir=str(tmp_path))

    assert agent.output_dir == str(tmp_path)
    assert agent.output_file_name("ppo_checkpoint_1") == "ppo_checkpoint_1"


def test_output_subdir_is_created_and_used(tmp_path):
    """
    With a subfolder, run-specific outputs go to model_dir/<subdir>.
    """

    agent = SinglePPOAgent(environment=_DummyEnv(), model_dir=str(tmp_path))
    agent.set_output_subdir("blk")

    assert (tmp_path / "blk").is_dir()
    assert agent.output_dir == os.path.join(str(tmp_path), "blk")
    assert agent.output_file_name("ppo_checkpoint_1") == os.path.join("blk", "ppo_checkpoint_1")

    agent.set_output_subdir(None)
    assert agent.output_dir == str(tmp_path)


@pytest.mark.parametrize("subdir", ["", " ", ".", "..", "a/b", "../x", "a\\b"])
def test_output_subdir_rejects_non_plain_names(tmp_path, subdir):
    """
    Only a single plain folder name is accepted.
    """

    agent = SinglePPOAgent(environment=_DummyEnv(), model_dir=str(tmp_path))
    with pytest.raises(ValueError):
        agent.set_output_subdir(subdir)


def test_learn_writes_checkpoints_and_metrics_to_subdir(tmp_path, monkeypatch):
    """
    Periodic/final checkpoints and training_metrics.csv go to the subfolder,
    the Best_ policy stays in model_dir.
    """

    agent = SinglePPOAgent(environment=_DummyEnv(), model_dir=str(tmp_path), eval_frequency=1)
    agent.set_output_subdir("blk")
    agent.MINIMUM_ITERATIONS_FOR_BEST_SAVE = 0
    saved, dumped = _stub_learn(agent, monkeypatch)

    agent.learn(stop_condition=StopCondition(max_time_steps=3))

    assert "Best_blk_PPO_policy" in saved
    checkpoints = [name for name in saved if "ppo_checkpoint" in name]
    assert checkpoints and all(name.startswith(os.path.join("blk", "")) for name in checkpoints)
    assert dumped == [(os.path.join(str(tmp_path), "blk"), "training_metrics.csv")]


def test_learn_writes_interrupt_checkpoint_to_subdir(tmp_path, monkeypatch):
    """
    The Ctrl-C checkpoint goes to the subfolder too.
    """

    def _interrupt(_data):
        """
        Simulate Ctrl-C during the PPO update.
        """

        raise KeyboardInterrupt()

    agent = SinglePPOAgent(environment=_DummyEnv(), model_dir=str(tmp_path))
    agent.set_output_subdir("blk")
    saved, dumped = _stub_learn(agent, monkeypatch, train=_interrupt)

    with pytest.raises(KeyboardInterrupt):
        agent.learn(stop_condition=StopCondition(max_time_steps=3))

    assert saved == [os.path.join("blk", "ppo_interrupt_0")]
    assert dumped == []


def test_set_environment_applies_output_subdir_and_reuses_root_policy(tmp_path, monkeypatch):
    """
    AgentManager gives the new agent its subfolder; the reused policy is
    still looked up in model_dir.
    """

    monkeypatch.setattr("fw.scheduling.agent_manager.apply_profile", lambda *a, **k: None)
    manager = AgentManager(PPO_POLICY_NAME, profile=None, hyperparameters={"model_dir": str(tmp_path)})
    manager.set_environment(env=_DummyEnv(), output_subdir="first")
    manager.agent.save_policy(f"first_{PPO_POLICY_NAME}_policy")

    loaded = []
    monkeypatch.setattr(SinglePPOAgent, "load_policy", lambda self, name: loaded.append(name))
    manager.set_environment(env=_DummyEnv(), policy_strategy=PolicyStrategy.REUSE_OTHER_POLICY,
                            other_env_name="first", output_subdir="second")

    assert (tmp_path / f"first_{PPO_POLICY_NAME}_policy.zip").is_file()
    assert loaded == [f"first_{PPO_POLICY_NAME}_policy"]
    assert manager.agent.output_dir == os.path.join(str(tmp_path), "second")


def test_missing_policy_for_visualize_leaves_no_output_folder(tmp_path, monkeypatch):
    """
    A visualize block whose policy is missing fails before its folder is created.
    """

    monkeypatch.setattr("fw.scheduling.agent_manager.apply_profile", lambda *a, **k: None)
    manager = AgentManager(PPO_POLICY_NAME, profile=None, hyperparameters={"model_dir": str(tmp_path)})

    with pytest.raises(FileNotFoundError):
        manager.set_environment(env=_DummyEnv(), policy_strategy=PolicyStrategy.REUSE_OTHER_POLICY,
                                other_env_name="missing", require_policy=True, output_subdir="viz")

    assert not (tmp_path / "viz").exists()


def test_visualize_saves_plots_in_output_dir(tmp_path, monkeypatch):
    """
    visualize_trained_agent renders its figures into agent.output_dir.
    """

    import fw.scheduling.agent_manager as agent_manager_module

    episode = SimpleNamespace(path=[(0.0, 0.0)])
    run_data = SimpleNamespace(ships={"ship": [episode]})
    renders = []
    monkeypatch.setattr(agent_manager_module.CollisionAnnotator, "annotate", lambda self, _data: None)
    monkeypatch.setattr(agent_manager_module.MetricsComputer, "compute", staticmethod(lambda _data: {}))
    monkeypatch.setattr(agent_manager_module.MatplotlibRenderer, "render",
                        staticmethod(lambda *_a, save_path, **_k: renders.append(save_path)))
    monkeypatch.setattr(agent_manager_module, "EpisodeCollector",
                        type("_Collector", (), {"__init__": lambda self, *a, **k: None,
                                                "collect": lambda self, **_k: run_data}))

    output_dir = tmp_path / "blk"
    manager = AgentManager(PPO_POLICY_NAME, profile=None, hyperparameters={})
    manager._agent = SimpleNamespace(model_dir=str(tmp_path), output_dir=str(output_dir),
                                     set_policy_eval=lambda: None, get_env=lambda: SimpleNamespace())

    manager.visualize_trained_agent(num_episodes=1, show_plots=False)

    assert renders == [f"{output_dir}/image.png"]
    assert output_dir.is_dir()


@pytest.mark.parametrize("action, transfer, expected", [
    ("train", TransferSpec(), "b"),
    ("visualize", TransferSpec(), "b_visualize"),
    ("visualize", TransferSpec(external_training_name="b"), "b_visualize"),
    ("visualize", TransferSpec(policy_strategy="REUSE_OTHER_POLICY"), "b_visualize"),
    ("visualize", TransferSpec(from_block="a"), "b"),
    ("visualize", TransferSpec(external_training_name="other"), "b"),
])
def test_block_output_subdir(action, transfer, expected):
    """
    Visualize blocks loading the policy named after themselves get their own folder.
    """

    block = PipelineBlock(name="b", env_type="PySimEnv", action=action, transfer=transfer)
    assert block_output_subdir(block) == expected


def test_validate_block_names_rejects_case_and_folder_clashes():
    """
    Names and output folders must be unique ignoring case.
    """

    with pytest.raises(ValueError, match="unique, ignoring case"):
        validate_block_names([PipelineBlock(name="A", env_type="PySimEnv"),
                              PipelineBlock(name="a", env_type="PySimEnv")])
    with pytest.raises(ValueError, match="output folder 'b_visualize'"):
        validate_block_names([PipelineBlock(name="b_visualize", env_type="PySimEnv"),
                              PipelineBlock(name="b", env_type="PySimEnv", action="visualize")])
    validate_block_names([PipelineBlock(name="b", env_type="PySimEnv"),
                          PipelineBlock(name="v", env_type="PySimEnv", action="visualize",
                                        transfer=TransferSpec(from_block="b"))])


def test_preflight_checks_names_of_blocks_built_in_python():
    """
    Blocks that bypass run_loader get the same name check before any block runs.
    """

    with pytest.raises(ValueError, match="reserved device name"):
        PipelineScheduler._preflight([PipelineBlock(name="aux", env_type="PySimEnv")])


@pytest.mark.parametrize("name", ["a/b", "a\\b", "x:y", "q?", "trail.", " lead", "trail ", "CON", "aux",
                                  "nul.txt", "COM1", "lpt9", "a\tb", "a\nb"])
def test_pipeline_rejects_block_names_unusable_as_folder(name):
    """
    Block names become folder names, so names invalid as a folder fail at load time.
    """

    from fw.scheduling.run_loader import PipelineSpecError, _parse_spec

    with pytest.raises(PipelineSpecError, match="file and folder name"):
        _parse_spec({"blocks": [{"name": name, "env_type": "PySimEnv"}]}, "test")


@pytest.mark.parametrize("name", ["base_train", "run 1 (fossen)", "console", "COM10", "auxiliary"])
def test_pipeline_accepts_ordinary_block_names(name):
    """
    Ordinary names, including ones that only start like a device name, are accepted.
    """

    from fw.scheduling.run_loader import _parse_spec

    assert _parse_spec({"blocks": [{"name": name, "env_type": "PySimEnv"}]}, "test").blocks[0].name == name


def test_pipeline_rejects_block_names_differing_only_in_case():
    """
    "A" and "a" would share a folder on macOS and Windows.
    """

    from fw.scheduling.run_loader import PipelineSpecError, _parse_spec

    with pytest.raises(PipelineSpecError, match="ignoring case"):
        _parse_spec({"blocks": [{"name": "A", "env_type": "PySimEnv"},
                                {"name": "a", "env_type": "PySimEnv"}]}, "test")
