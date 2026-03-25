import os
import re
import csv
import importlib
from datetime import datetime
from datetime import datetime

from enum import Enum
from pathlib import Path
from typing import Dict, List, Any
from fw.agent_manager import AgentManager
from fw.agent_manager import PolicyStrategy
from fw.config import get_config_parameters, get_hyperparameters
from fw.env_factory import (
    resolve_environment_name,
    py_sim_env_name,
    fh_sim_env_name,
    airsim_env_name,
    lunar_env_name,
    py_sim_polar_env_name,
    PySimEnvironment,
    FhSimEnvironment,
    AirSimEnvironment,
    LunarEnvironment,
    PySimPolarEnvironment,
)
from contextlib import contextmanager

# logging.basicConfig(level=logging.INFO)  # Configure logging once, possibly in main script.


class FilenameStyle(Enum):
    """Enumeration of supported filename styles."""
    SNAKE = 'snake'
    LOWER = 'lower'
    KEBAB = 'kebab'


def class_name_to_filename(class_name: str, style: FilenameStyle) -> str:
    """Converts a class name to a filename based on the given style.

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
    """Finds the filename corresponding to a class name based on existing files.

    Tries different naming styles (snake_case, lowercase, kebab-case) until it finds a match.

    Args:
        class_name (str): The name of the class to search for.
        directory (str, optional): The directory to search in. Defaults to the current directory.

    Returns:
        str: The relative path to the matching file.

    Raises:
        FileNotFoundError: If no matching file is found using any style.
    """
    for style in FilenameStyle:
        filename = class_name_to_filename(class_name, style)
        file_path = os.path.join(directory, filename)
        if os.path.isfile(file_path):
            return filename  # Match found in the given directory
        elif os.path.isfile(f"fw/{file_path}"):
            return f"fw/{filename}"  # Match found in the 'fw/' subdirectory

    raise FileNotFoundError(
        f"No file found for class '{class_name}' in {directory}. "
        f"Tried styles: {[class_name_to_filename(class_name, s) for s in FilenameStyle]}"
    )


class Trainer:
    """Handles the training of an agent across multiple environments.

    Responsible for creating the agent manager, initializing the scheduler, and managing the training loop.
    """

    def __init__(self, agent_type: str, envs, scheduler_type: str):
        """Initializes the Trainer.

        Args:
            agent_type (str): The type of agent to create (e.g., 'PPO', 'DQN').
            envs (Gym.Env): List of environments for the agent to train in.
            scheduler_type (str): The name of the scheduler class to use.

        Raises:
            RuntimeError: If the scheduler class cannot be found or initialized.
            TypeError: If the scheduler does not have required attributes.
        """
        self._agent_type = agent_type
        self._envs = envs
        self._config = get_config_parameters()
        self._agent_mgr = AgentManager(agent_type, get_hyperparameters(agent_type))

        # Dynamically load the scheduler based on its name
        try:
            scheduler_file = Path(find_existing_filename(scheduler_type))
            scheduler_module = importlib.import_module(f"fw.{scheduler_file.stem}")
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
        self.prev_render_mode = None


    @contextmanager
    def render(self, env):
        """Render.

        Args:
            env: Description.
        """
        if hasattr(env, "render_mode"):
            try:
                self.prev_render_mode = env.render_mode
                env.render_mode = "human"
                yield
            finally:
                env.render_mode = self.prev_render_mode
                self.prev_render_mode = None


    def train(self, configuration: dict = None) -> dict:
        """Starts the training process.

        Args:
            configuration (dict, optional): Additional configuration parameters
                for training, such as number of steps per environment.

        Returns:
            dict: Training metrics collected during execution.
        """
        metrics = self._scheduler.execute(configuration)

        last_env = self._envs[-1] if self._envs else None
        env_type = last_env.gym_env.type_name if last_env else None
        env_metrics = metrics.get(env_type) if env_type else None

        if env_metrics and env_metrics.get("status") == "success":
            try:
                summaries = self.visualize([last_env], collect_evaluation_summary=True)
                if summaries and env_type in summaries:
                    env_metrics["evaluation_summary"] = summaries[env_type]
            except AttributeError as e:
                print(e)
        else:
            print(f"No successful training found for environment '{env_type}' to visualize.")

        return metrics
    

    def visualize(self, envs, airsim_bridge=None, collect_evaluation_summary: bool = False):
        """Visualize.

        Args:
            envs: Description.
            airsim_bridge: Description.
            collect_evaluation_summary: Whether to return final evaluation metrics.
        """
        summaries = {} if collect_evaluation_summary else None
        if envs is not None:
            # Start shared figure
            self._agent_mgr.begin_overlay_plot()

            for env_idx, env in enumerate(envs):
                gym_env = env.gym_env
                try:
                    with self.render(gym_env):
                        if hasattr(gym_env, "phys_sims"):
                            if hasattr(gym_env.phys_sims[0], "_dynamics"):
                                print(f"Visualization of {gym_env.ship_model.name} in a {gym_env.dynamics_model.name}-based simulator.")
                            elif hasattr(gym_env.phys_sims[0], "_engine"):
                                print(f"Visualization of {gym_env.ship_model.name} in an {gym_env.phys_sims[0]._engine} simulator.")

                        policy_name = f"Best_{gym_env.type_name}" if self._config[
                            "visualize_best_trained"] else gym_env.type_name

                        self._agent_mgr.set_environment(
                            env=gym_env,
                            policy_strategy=PolicyStrategy.REUSE_OTHER_POLICY,
                            other_env_name=policy_name,
                        )

                        eval_summary = self._agent_mgr.visualize_trained_agent(
                            num_episodes=1,
                            live_animation=self._config["live_animation"],
                            airsim_bridge=airsim_bridge,
                            return_summary=collect_evaluation_summary,
                        )
                        if collect_evaluation_summary and eval_summary is not None:
                            summaries[gym_env.type_name] = eval_summary
                except RuntimeError as e:
                    print(f"Could not visualize {gym_env.type_name}. {e}")

            # Finalize shared figure
            self._agent_mgr.end_overlay_plot()
        else:
            print("No data found to visualize.")

        return summaries


    @staticmethod
    def _build_eval_environment(canonical_name: str, config: Dict[str, Any]):
        """Build a fresh environment wrapper for reward evaluation.

        This bypasses the factory cache so each reward configuration gets
        its own independent environment instance.
        """
        if canonical_name == py_sim_env_name:
            return PySimEnvironment(canonical_name, config)
        if canonical_name == fh_sim_env_name:
            return FhSimEnvironment(canonical_name, config)
        if canonical_name == airsim_env_name:
            return AirSimEnvironment(canonical_name, config)
        if canonical_name == lunar_env_name:
            return LunarEnvironment(canonical_name, config)
        if canonical_name == py_sim_polar_env_name:
            return PySimPolarEnvironment(canonical_name, config)
        raise ValueError(f"Unsupported environment '{canonical_name}' for reward evaluation")
    

    def evaluate_reward(
        self,
        reward_configs: List[Dict[str, Any]],
        training_configuration: Dict[str, Any] = None,
        verbose: bool = True
    ) -> Dict[int, Dict[str, Any]]:
        """Evaluate and compare different reward configurations.

        Trains a separate, independent policy for each reward configuration.
        Stores and displays metrics for comparison.

        Args:
            reward_configs: List of environment configuration dicts. Each dict should have:
                - "type": Environment type (e.g., "PySimEnvironment")
                - "reward_name": Reward profile name (e.g., "default", "heading_only")
                - "reward_weights": Dict of component weights (optional)
                - Any other environment-specific config

            training_configuration: Dict of Stop Conditions keyed by environment type.
                Example:
                {
                    "PySimEnvironment": StopCondition(...),
                    "FhSimEnvironment": StopCondition(...)
                }

            verbose: Print progress information.

        Returns:
            Dict mapping config index to:
            {
                "config": original configuration,
                "env_type": environment type,
                "reward_name": reward profile name,
                "reward_weights": weights used,
                "metrics": training metrics dict,
                "status": success/failure status
            }
        """
        if not reward_configs:
            raise ValueError("reward_configs must not be empty")

        if training_configuration is None:
            training_configuration = {}

        if verbose:
            print(f"\n{'='*80}")
            print(f"Starting Reward Evaluation: {len(reward_configs)} configurations")
            print(f"{'='*80}\n")

        results = {}
        evaluation_order = []

        for config_idx, config in enumerate(reward_configs):
            try:
                # Extract environment type
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

                # Build a fresh environment per configuration (no factory cache reuse)
                env = self._build_eval_environment(canonical_name, config)

                # Create independent trainer for this config
                trainer = Trainer(self._agent_type, [env], "SimpleScheduler")

                # Train
                if verbose:
                    print(f"    Starting training...")

                metrics = trainer.train(training_configuration)

                trajectory_file = self._save_evaluation_trajectory_snapshot(trainer, config_idx + 1)
                if verbose and trajectory_file is not None:
                    print(f"    Saved trajectory: {trajectory_file}")

                # Store results
                results[config_idx] = {
                    "config": config,
                    "env_type": canonical_name,
                    "reward_name": reward_name,
                    "reward_weights": reward_weights,
                    "metrics": metrics,
                    "trajectory_file": trajectory_file,
                    "status": "success"
                }

                if verbose:
                    print(f"    ✓ Training completed")

                evaluation_order.append(config_idx)

            except Exception as e:
                if verbose:
                    print(f"    ✗ Training failed: {e}")

                results[config_idx] = {
                    "config": config,
                    "env_type": config.get("type", "Unknown"),
                    "reward_name": config.get("reward_name", "unknown"),
                    "reward_weights": config.get("reward_weights", None),
                    "metrics": None,
                    "trajectory_file": None,
                    "status": f"failed: {str(e)}"
                }

        if verbose:
            self._print_comparison_summary(results, evaluation_order)

        return results


    def _print_comparison_summary(self, results: Dict[int, Dict[str, Any]], evaluation_order: List[int]) -> None:
        """Print a summary comparing all evaluated configurations.

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

        # Print detailed results for each configuration
        for idx in evaluation_order:
            result = results[idx]
            print(f"Configuration {idx}: {result['env_type']}")
            print(f"  Reward Profile: {result['reward_name']}")

            if result['reward_weights']:
                print("  Weights:")
                for key, value in result['reward_weights'].items():
                    print(f"    - {key}: {value}")

            if result['status'] == "success":
                print("  Status: ✓ SUCCESS")
                self._print_metrics(result['metrics'])
            else:
                print(f"  Status: ✗ FAILED ({result['status']})")

            print("-" * 80)


    def _print_metrics(self, metrics: Dict[str, Any]) -> None:
        """Pretty-print training metrics.

        Args:
            metrics: Metrics dictionary from training.
        """
        if not metrics:
            print("  Metrics: None")
            return

        print("  Metrics:")
        for env_type, env_metrics in metrics.items():
            if isinstance(env_metrics, dict):
                for key, value in env_metrics.items():
                    if isinstance(value, float):
                        print(f"    {key}: {value:.4f}")
                    else:
                        print(f"    {key}: {value}")


    @staticmethod
    def _save_evaluation_trajectory_snapshot(eval_trainer: "Trainer", sequence_index: int) -> str | None:
        """Rename evaluate_reward trajectory output to trajectory_n_time.png.

        Args:
            eval_trainer: Trainer instance that just completed one evaluation run.
            sequence_index: 1-based index in the evaluation sequence.

        Returns:
            Saved file path or None when no trajectory file exists.
        """
        agent = getattr(eval_trainer._agent_mgr, "agent", None)
        model_dir = getattr(agent, "model_dir", None) if agent is not None else None
        source_path = os.path.join(model_dir, "trajectory.png") if model_dir else "trajectory.png"

        if not os.path.exists(source_path):
            return None

        time_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        target_name = f"trajectory_{sequence_index}_{time_stamp}.png"
        target_path = os.path.join(model_dir, target_name) if model_dir else target_name

        os.replace(source_path, target_path)
        return target_path


    def export_evaluation_results(self, results: Dict[int, Dict[str, Any]], filepath: str) -> None:
        """Export evaluation results to CSV for easy comparison.

        Args:
            results: Results dictionary from evaluate_reward.
            filepath: Path to write CSV file.
        """
        if not results:
            print("No results to export.")
            return

        try:
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
                "Mean LOS Guidance Heading Error (mHE)",
            ]

            weight_keys = sorted({
                weight_key
                for result in results.values()
                for weight_key in (result.get("reward_weights") or {}).keys()
            })

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

                    # Extract final evaluation summary from train metrics.
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
        except Exception as e:
            print(f"Failed to export results: {e}")

