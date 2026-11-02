import os
import re
import csv
import copy
import logging
import importlib

from enum import Enum
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
from fw.configuration.env_factory import (
    resolve_environment_name,
    py_sim_env_name,
    fh_sim_env_name,
    airsim_env_name,
    lunar_env_name,
    PySimEnvironment,
    FhSimEnvironment,
    AirSimEnvironment,
    LunarEnvironment,
    FhSimSyncEnvironment, sync_sim_env_name,
)
from fw.scheduling.agent_manager import AgentManager
from fw.scheduling.agent_manager_simple import AgentManagerSimple
from fw.configuration.config import get_config_parameters, get_hyperparameters, VisualizationMode, PolicyStrategy
from fw.configuration.config_profile import VesselProfile

logger = logging.getLogger(__name__)


class FilenameStyle(Enum):
    """
    Enumeration of supported filename styles.
    """

    SNAKE = 'snake'
    LOWER = 'lower'
    KEBAB = 'kebab'


def class_name_to_filename(class_name: str, style: FilenameStyle) -> str:
    """
    Converts a class name to a filename based on the given style.

    Args:
        class_name (str): The name of the class to convert.
        style (FilenameStyle): The desired filename style (snake_case, lowercase, kebab-case).

    Returns:
        str: The resulting filename.

    Raises:
        ValueError: If an unknown filename style is provided.
    """

    # Insert underscores before capital letters and lowercase the result
    pattern = r'(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])'
    split_name = re.sub(pattern, '_', class_name).lower()

    if style == FilenameStyle.SNAKE:
        filename = f"{split_name}.py"
    elif style == FilenameStyle.LOWER:
        filename = f"{split_name.replace('_', '')}.py"
    elif style == FilenameStyle.KEBAB:
        filename = f"{split_name.replace('_', '-')}.py"
    else:
        raise ValueError(f"Unknown style: {style}")

    return filename


def find_existing_filename(class_name: str, directory: str = ".") -> str:
    """
    Find the file name of a class by trying the naming styles (snake_case,
    lowercase, kebab-case).

    Per style, it looks in `directory`, `<directory>/fw/` (both cwd-relative,
    for older layouts) and this module's folder `fw/scheduling/`, where the
    scheduler modules live (independent of the launch directory).

    Args:
        class_name (str): The name of the class to search for.
        directory (str, optional): Directory to search first; defaults to the
            current directory.

    Returns:
        str: The matching file name without directory (the caller imports
            it from `fw.scheduling.`).

    Raises:
        FileNotFoundError: If no matching file is found using any style.
    """

    module_dir = os.path.dirname(os.path.abspath(__file__))

    for style in FilenameStyle:
        filename = class_name_to_filename(class_name, style)

        candidates = [
            os.path.join(directory, filename),
            os.path.join(directory, "fw", filename),
            os.path.join(module_dir, filename),
        ]
        for candidate in candidates:
            if os.path.isfile(candidate):
                return filename

    raise FileNotFoundError(
        f"No file found for class '{class_name}' in '{directory}' or "
        f"'{module_dir}'. Tried styles: "
        f"{[class_name_to_filename(class_name, s) for s in FilenameStyle]}"
    )


class Trainer:
    """
    Handles the training of an agent across multiple environments.

    Creates the agent manager and the scheduler, and runs the training loop.
    """

    def __init__(self, agent_type: str, envs, profile: VesselProfile, scheduler_type: str,
                 config: dict = None):
        """
        Initialize the Trainer.

        Args:
            agent_type (str): The type of agent to create (e.g., 'PPO', 'DQN').
            envs (Gym.Env): List of environments for the agent to train in.
            profile (VesselProfile): Profile that will be used for training or visualization.
            scheduler_type (str): The name of the scheduler class to use.
            config (dict, optional): Pre-built config from
                `get_config_parameters()`; None builds it here. Pass it when
                already built (e.g. main.py) to avoid applying the profile twice.

        Raises:
            RuntimeError: If the scheduler class cannot be found or initialized.
            TypeError: If the scheduler does not have required attributes.
        """

        self._agent_type = agent_type
        self._envs = envs
        self._profile = profile
        self._config = config if config is not None else get_config_parameters()
        self._hyperparameter_env_name = self._resolve_hyperparameter_env_name()

        if self._config["viz_mode"] == VisualizationMode.SIMPLE:
            self._agent_mgr = AgentManagerSimple(
                agent_type,
                profile,
                get_hyperparameters(agent_type, env_name=self._hyperparameter_env_name),
            )
        else:
            self._agent_mgr = AgentManager(
                agent_type,
                profile,
                get_hyperparameters(agent_type, env_name=self._hyperparameter_env_name),
            )

        # Dynamically load the scheduler based on its name
        try:
            scheduler_file = Path(find_existing_filename(scheduler_type))
            scheduler_module = importlib.import_module(f"fw.scheduling.{scheduler_file.stem}")
            scheduler_class = getattr(scheduler_module, scheduler_type)
            scheduler = scheduler_class(self._agent_mgr, self._envs)
        except ModuleNotFoundError:
            raise RuntimeError(f"Cannot create unknown scheduler class '{scheduler_type}'.")
        except Exception as e:
            raise RuntimeError(str(e))

        # Validate that the scheduler has required interface
        if not hasattr(scheduler, 'envs') or not callable(getattr(scheduler, 'execute', None)):
            raise TypeError("Scheduler must have an 'envs' attribute and an 'execute' method.")

        self._scheduler = scheduler

    def _resolve_hyperparameter_env_name(self):
        """
        Return a shared environment name when all selected envs match.

        Mixed environment families give None (shared PPO defaults), so one env
        does not change the tuning of the others.
        """

        env_names = []
        for env in self._envs:
            type_name = getattr(env, "type_name", None)
            if type_name is None:
                type_name = getattr(getattr(env, "gym_env", None), "type_name", None)
            if type_name:
                env_names.append(type_name)

        if env_names and len(set(env_names)) == 1:
            return env_names[0]

        return None


    def train(self, configuration: dict = None, show_plots: bool = True,
              evaluate: Optional[bool] = None) -> dict:
        """
        Starts the training process.

        Args:
            configuration (dict, optional): Additional configuration parameters
                for training, such as number of steps per environment.
            show_plots (bool, optional): Show interactive figures. Also decides
                whether the post-training evaluation (rollout, metrics,
                figures) runs at all when `evaluate` is None. Default True.
            evaluate (bool, optional): Run the post-training evaluation episode
                and store `evaluation_summary` in the metrics. None follows
                `show_plots`; `evaluate_reward` passes True with
                show_plots=False. Skipped for AirSimEnv.

        Returns:
            dict: Training metrics collected during execution.
        """

        metrics = self._scheduler.execute(configuration)

        if evaluate is None:
            evaluate = show_plots
        if not evaluate:
            return metrics

        last_env = self._envs[-1] if self._envs else None
        env_type = str(last_env) if last_env else None
        env_metrics = metrics.get(env_type) if env_type else None

        if env_metrics and env_metrics.get("status") == "success":
            try:
                if env_type == "AirSimEnv":
                    logger.info("Skipping automatic post-training visualization for AirSimEnv.")
                else:
                    summaries = self.visualize(
                        [last_env],
                        collect_evaluation_summary=True,
                        show_plots=show_plots,
                    )
                    if summaries:
                        env_metrics["evaluation_summary"] = (
                            summaries.get(env_type)
                            or next(iter(summaries.values()))
                        )
            except AttributeError as e:
                logger.warning(e)
        else:
            logger.warning("No successful training found for environment '%s' to visualize.", env_type)

        return metrics
    

    def visualize(
        self,
        envs,
        airsim_bridge=None,
        collect_evaluation_summary: bool = False,
        show_plots: bool = True,
    ):
        """
        Run the trained agent(s) and render/collect evaluation results.

        Args:
            envs: Iterable of trained environment wrappers (each exposing a
                  `.gym_env` and `.agent_mgr`) to visualize/evaluate, or None
                  to skip visualization entirely.
            airsim_bridge: Optional AirSim coordinate-transform bridge used to
                  mirror ship pose into UE5 during evaluation; ignored if None.
            collect_evaluation_summary: Whether to return final evaluation metrics.
            show_plots: Whether to show interactive matplotlib windows.
        """

        summaries = {} if collect_evaluation_summary else None

        if envs is not None:
            for env_idx, env in enumerate(envs):
                gym_env = env.gym_env
                try:
                    sim = getattr(gym_env, "phys_sim", None)
                    if sim is not None:
                        sim_name = (
                            f"{gym_env.dynamics_model.name}-based simulator"
                            if hasattr(sim, "_dynamics")
                            else None
                        )

                        if sim_name:
                            logger.info("Visualization of %s in a %s.", gym_env.ship_model.name, sim_name)

                    policy_name = f"Best_{gym_env.training_name}" if self._config[
                        "visualize_best_trained"] else gym_env.training_name

                    self._agent_mgr.set_environment(
                        env=gym_env,
                        policy_strategy=PolicyStrategy.REUSE_OTHER_POLICY,
                        other_env_name=policy_name,
                    )

                    env_summary = self._agent_mgr.visualize_trained_agent(
                        num_episodes=1,
                        live_animation=self._config["live_animation"],
                        airsim_bridge=airsim_bridge,
                        return_summary=collect_evaluation_summary,
                        show_plots=show_plots,
                    )
                    if collect_evaluation_summary and env_summary is not None:
                        if isinstance(env_summary, dict):
                            if gym_env.type_name in env_summary:
                                # Look up by type_name (the gym env's key), store
                                # under training_name (what train() looks up).
                                summaries[gym_env.training_name] = env_summary[gym_env.type_name]
                            else:
                                # For composite envs (e.g. MultiSimEnv), collect summaries for all
                                # sub-environments whose keys appear in env_summary.
                                sub_envs = getattr(gym_env, "sub_environments", None)
                                if sub_envs is not None:
                                    matched = {
                                        name: env_summary[name]
                                        for name in sub_envs
                                        if name in env_summary
                                    }
                                    if matched:
                                        summaries[gym_env.training_name] = matched
                                    else:
                                        logger.warning(
                                            "visualize(): env_summary dict has no key for "
                                            "'%s' or its sub-environments (%s). Available keys: "
                                            "%s. Skipping summary for this env.",
                                            gym_env.type_name, list(sub_envs.keys()), list(env_summary.keys()),
                                        )
                                else:
                                    logger.warning(
                                        "visualize(): env_summary dict has no key for "
                                        "'%s'. Available keys: %s. Skipping summary for this env.",
                                        gym_env.type_name, list(env_summary.keys()),
                                    )
                        else:
                            summaries[gym_env.training_name] = env_summary
                except RuntimeError as e:
                    logger.warning("Could not visualize %s. %s", gym_env.training_name, e)
        else:
            logger.warning("No data found to visualize.")

        if collect_evaluation_summary:
            return summaries

        return None

    @staticmethod
    def _build_eval_environment(canonical_name: str, config: Dict[str, Any]):
        """
        Build a fresh environment wrapper for reward evaluation.

        Bypasses the factory cache so each reward configuration is isolated.
        """

        if canonical_name == py_sim_env_name:
            return PySimEnvironment(canonical_name, config)
        if canonical_name == fh_sim_env_name:
            return FhSimEnvironment(canonical_name, config)
        if canonical_name == sync_sim_env_name:
            return FhSimSyncEnvironment(canonical_name, config)
        if canonical_name == airsim_env_name:
            return AirSimEnvironment(canonical_name, config)
        if canonical_name == lunar_env_name:
            return LunarEnvironment(canonical_name, config)
        raise ValueError(f"Unsupported environment '{canonical_name}' for reward evaluation")
    

    def evaluate_reward(
        self,
        reward_configs: List[Dict[str, Any]],
        training_configuration: Dict[str, Any] = None,
        verbose: bool = True,
    ) -> Dict[int, Dict[str, Any]]:
        """
        Evaluate and compare different reward configurations.

        Trains an independent policy per reward configuration, sequentially.

        Args:
            reward_configs (List[Dict[str, Any]]): List of environment configuration dicts.
                Each dict should have:
                    - "type": Environment type (e.g., "PySimEnvironment")
                    - "reward_name": Reward profile name (e.g., "default", "heading_only")
                    - "reward_weights": Dict of component weights (optional)
            training_configuration (Dict[str, Any], optional): Dict of stop conditions
                keyed by environment type. Example:
                    {
                        "PySimEnvironment": StopCondition(...),
                    }
            verbose (bool, optional): If True, prints progress information.

        Returns:
            Dict[int, Dict[str, Any]]: Mapping of config index to results:
                {
                    "config": original configuration,
                    "env_type": environment type,
                    "reward_name": reward profile name,
                    "reward_weights": weights used,
                    "metrics": training metrics dict,
                    "trajectory_file": saved trajectory image or None,
                    "status": "success" or "failed: <error>"
                }
        """

        # Enforce explicit configs to avoid silent no-op evaluations.
        if not reward_configs:
            raise ValueError("reward_configs must not be empty")

        # Allow caller to pass only reward configs and use default stop conditions.
        if training_configuration is None:
            training_configuration = {}

        if verbose:
            print(f"\n{'='*80}")
            print(f"Starting Reward Evaluation: {len(reward_configs)} configurations")
            print(f"{'='*80}\n")

        results = {}
        # Indices of successful configs.
        evaluation_order = []

        def _evaluate_single_config(config_idx: int, config: Dict[str, Any]) -> tuple[int, Dict[str, Any], bool]:
            """
            Train one reward configuration in an isolated trainer instance.
            """

            try:
                if "type" not in config:
                    raise ValueError("Configuration missing required 'type' key")

                env_type = config["type"]
                canonical_name = resolve_environment_name(env_type)
                reward_name = config.get("reward_name", "default")
                reward_weights = config.get("reward_weights", None)

                if verbose:
                    print(f"\n[{config_idx + 1}] Environment: {canonical_name}")
                    print(f"    Reward Profile: {reward_name}")
                    if reward_weights:
                        for key, value in reward_weights.items():
                            print(f"      - {key}: {value}")
                    print("    Starting training...")

                env = self._build_eval_environment(canonical_name, config)

                trainer = Trainer(self._agent_type, [env], self._profile, "SimpleScheduler")

                # Deep copy, so StopCondition counters do not carry over to later configs.
                isolated_configuration = copy.deepcopy(training_configuration)

                # No interactive plots, but run the evaluation episode: the CSV
                # export reads its evaluation_summary and the snapshot below
                # needs the image it renders.
                metrics = trainer.train(isolated_configuration, show_plots=False, evaluate=True)
                # Snapshot the trajectory image with a unique name per config.
                trajectory_file = self._save_evaluation_trajectory_snapshot(trainer, reward_weights)

                if verbose:
                    if trajectory_file is not None:
                        print(f"    Saved trajectory: {trajectory_file}")
                    print("    Training completed")

                return config_idx, {
                    "config": config,
                    "env_type": canonical_name,
                    "reward_name": reward_name,
                    "reward_weights": reward_weights,
                    "metrics": metrics,
                    "trajectory_file": trajectory_file,
                    "status": "success"
                }, True
            except Exception as e:
                if verbose:
                    print(f"    Training failed: {e}")
                logger.error("Reward evaluation config %d failed: %s", config_idx, e)

                # Same shape on failure, so downstream code can continue.
                return config_idx, {
                    "config": config,
                    "env_type": config.get("type", "Unknown"),
                    "reward_name": config.get("reward_name", "unknown"),
                    "reward_weights": config.get("reward_weights", None),
                    "metrics": None,
                    "trajectory_file": None,
                    "status": f"failed: {str(e)}"
                }, False

        # Sequential: one environment execution flow at a time.
        job_results = []
        for conf_idx, conf in enumerate(reward_configs):
            job_results.append(_evaluate_single_config(conf_idx, conf))

        for idx, result, is_success in job_results:
            results[idx] = result
            if is_success:
                evaluation_order.append(idx)

        evaluation_order.sort()

        if verbose:
            self._print_comparison_summary(results, evaluation_order)

        return results


    def _print_comparison_summary(self, results: Dict[int, Dict[str, Any]], evaluation_order: List[int]) -> None:
        """
        Print a summary comparing all evaluated configurations.

        Args:
            results: Results dictionary from evaluate_reward.
            evaluation_order: Order in which configs were evaluated.
        """

        print(f"\n{'='*80}")
        print("REWARD CONFIGURATION COMPARISON SUMMARY")
        print(f"{'='*80}\n")

        if not results:
            print("No configurations evaluated.")
            return

        for idx in evaluation_order:
            result = results[idx]
            print(f"Configuration {idx}: {result['env_type']}")
            print(f"  Reward Profile: {result['reward_name']}")

            if result['reward_weights']:
                print("  Weights:")
                for key, value in result['reward_weights'].items():
                    print(f"    - {key}: {value}")

            if result['status'] == "success":
                print("  Status: SUCCESS")
                self._print_metrics(result['metrics'])
            else:
                print(f"  Status: FAILED ({result['status']})")

            print("-" * 80)

        # Failed configurations get their own section, never omitted.
        failed_ids = [i for i in sorted(results) if i not in evaluation_order]
        if failed_ids:
            print(f"\n{'='*80}")
            print(f"FAILED CONFIGURATIONS ({len(failed_ids)})")
            print(f"{'='*80}\n")
            for idx in failed_ids:
                result = results[idx]
                print(f"Configuration {idx}: {result.get('env_type', 'Unknown')}")
                print(f"  Reward Profile: {result.get('reward_name', 'unknown')}")
                print(f"  Status: {result.get('status', 'failed')}")
                print("-" * 80)


    @staticmethod
    def _print_metrics(metrics: Dict[str, Any]) -> None:
        """
        Pretty-print training metrics.

        Args:
            metrics: Metrics dictionary from training.
        """

        if not metrics:
            print("  Metrics: None")
            return

        print("  Metrics:")
        for env_type, env_metrics in metrics.items():
            if isinstance(env_metrics, dict):
                print(f"    [{env_type}]")
                for key, value in env_metrics.items():
                    if isinstance(value, float):
                        print(f"      {key}: {value:.4f}")
                    else:
                        print(f"      {key}: {value}")


    @staticmethod
    def _save_evaluation_trajectory_snapshot(eval_trainer: "Trainer", reward_weights: dict | None) -> str | None:
        """
        Save evaluate_reward trajectory output as trajectory_{weights}.png.

        Args:
            eval_trainer: Trainer instance that just completed one evaluation run.
            reward_weights: Dictionary of reward weights used in this configuration (or None for default).

        Returns:
            Saved file path or None when no trajectory file exists.
        """

        agent = getattr(eval_trainer._agent_mgr, "agent", None)
        model_dir = getattr(agent, "model_dir", None) if agent is not None else None
        if not model_dir:
            model_dir = "models"
        os.makedirs(model_dir, exist_ok=True)
        # Candidate images in the model dir or cwd.
        candidates = [
            os.path.join(model_dir, "trajectory.png"),
            os.path.join(model_dir, "image_trajectory.png"),
            os.path.join(model_dir, "image.png"),
            os.path.join(model_dir, "image_metrics.png"),
            "trajectory.png",
            "image_trajectory.png",
            "image.png",
            "image_metrics.png",
        ]

        found = {}
        for p in candidates:
            if os.path.exists(p):
                name = os.path.basename(p)
                found[name] = p

        if not found:
            return None

        # Compact weights suffix (dots in values replaced).
        if reward_weights:
            weights_str = "_".join(
                f"{str(value).replace('.', '')}"
                for value in reward_weights.values()
            )
        else:
            weights_str = "default"

        def _move_with_suffix(src_path: str, prefix: str) -> str:
            """
            Move a file to `<prefix>_<weights>.png` (timestamp appended if the
            target exists) with os.replace.

            Args:
                src_path: Source file path to move.
                prefix: Filename prefix (e.g., 'trajectory', 'image_metrics').
            
            Returns:
                str: Final target file path after move operation.
            """

            target_name = f"{prefix}_{weights_str}.png"
            target_path = os.path.join(model_dir, target_name)
            if os.path.exists(target_path):
                time_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                target_name = f"{prefix}_{weights_str}_{time_stamp}.png"
                target_path = os.path.join(model_dir, target_name)
            os.replace(src_path, target_path)
            return target_path

        trajectory_target = None

        # If a canonical 'trajectory.png' or 'image_trajectory.png' exists, move it.
        if 'trajectory.png' in found:
            trajectory_target = _move_with_suffix(found['trajectory.png'], 'trajectory')
        elif 'image_trajectory.png' in found:
            trajectory_target = _move_with_suffix(found['image_trajectory.png'], 'image_trajectory')
        elif 'image.png' in found:
            # image.png may be a combined file: fall back to it as trajectory
            trajectory_target = _move_with_suffix(found['image.png'], 'trajectory')

        # Metrics image, if present
        if 'image_metrics.png' in found:
            _move_with_suffix(found['image_metrics.png'], 'image_metrics')

        return trajectory_target


    @staticmethod
    def export_evaluation_results(results: Dict[int, Dict[str, Any]], filepath: str) -> None:
        """
        Export evaluation results to CSV for easy comparison.

        Args:
            results: Results dictionary from evaluate_reward.
            filepath: Path to write CSV file.
        """

        if not results:
            logger.warning("No results to export.")
            return

        try:
            models_dir = Path("models")
            models_dir.mkdir(exist_ok=True)
            
            # Timestamped file name
            base_path = Path(filepath)
            stem = base_path.stem
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            new_filename = f"{stem}_{timestamp}.csv"
            filepath = str(models_dir / new_filename)
            
            metric_fields = [
                "Average reward",
                "Best episode reward",
                "Worst episode reward",
                "Mean Surge Velocity (mu)",
                "Maximum Surge Velocity (Mu)",
                "Mean Track Error Integral (mTEI)",
                "Maximum Track Error (MTE)",
                "Mean Rudder Angle (mRA)",
                "Maximum Rudder Angle (MRA)",
                "Mean Rudder Angle Change (mRAC)",
                "Steering Variance (vRA)",
                "Steering Sign-Change Frequency (sCF)",
                "Mean Thrust (mT)",
                "Maximum Thrust (MT)",
                "Mean Thrust Change (mTC)",
                "Mean LOS Guidance Heading Error (mHE)",
            ]

            # Weight columns in configuration insertion order.
            weight_keys = []
            for idx in sorted(results.keys()):
                weights = results[idx].get("reward_weights") or {}
                for weight_key in weights.keys():
                    if weight_key not in weight_keys:
                        weight_keys.append(weight_key)

            with open(filepath, 'w', newline='') as f:
                writer = csv.writer(f)

                # Header
                writer.writerow([
                    "Config Index",
                    "Environment Type",
                    "Reward Profile",
                    "Status",
                    *metric_fields,
                    *[f"Weight: {key}" for key in weight_keys],
                ])

                # Rows
                for idx in sorted(results.keys()):
                    result = results[idx]

                    # Final evaluation summary from the train metrics.
                    summary = {}
                    metrics = result.get("metrics") or {}
                    if isinstance(metrics, dict):
                        for env_metrics in metrics.values():
                            if isinstance(env_metrics, dict) and "evaluation_summary" in env_metrics:
                                summary = env_metrics.get("evaluation_summary") or {}
                                break

                    row_metrics = [
                        summary.get("average_reward"),
                        summary.get("best_episode_reward"),
                        summary.get("worst_episode_reward"),
                        summary.get("mu"),
                        summary.get("Mu"),
                        summary.get("mTEI"),
                        summary.get("MTE"),
                        summary.get("mRA"),
                        summary.get("MRA"),
                        summary.get("mRAC"),
                        summary.get("vRA"),
                        summary.get("sCF"),
                        summary.get("mT"),
                        summary.get("MT"),
                        summary.get("mTC"),
                        summary.get("mHE"),
                    ]

                    weights = result.get("reward_weights") or {}
                    row_weights = [weights.get(key) for key in weight_keys]

                    writer.writerow([
                        idx,
                        result['env_type'],
                        result['reward_name'],
                        result['status'],
                        *row_metrics,
                        *row_weights,
                    ])

            print(f"Results exported to {filepath}")
            logger.info("Results exported to %s", filepath)
        except Exception as e:
            logger.error("Failed to export results: %s", e)
