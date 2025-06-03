import os
import re
import importlib

from enum import Enum
from pathlib import Path
from fw.agent import Agent
from fw.agent import PolicyStrategy
from fw.config import get_config_parameters, get_hyperparameters
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

    Responsible for creating the agent, initializing the scheduler, and managing the training loop.
    """

    def __init__(self, model_type: str, envs, scheduler_type: str):
        """Initializes the Trainer.

        Args:
            model_type (str): The type of model to create (e.g., 'PPO', 'DQN').
            envs (Gym.Env): List of environments for the agent to train in.
            scheduler_type (str): The name of the scheduler class to use.

        Raises:
            RuntimeError: If the scheduler class cannot be found or initialized.
            TypeError: If the scheduler does not have required attributes.
        """
        self._envs = envs
        self._config = get_config_parameters()
        self._agent = Agent(model_type, get_hyperparameters(model_type))

        # Dynamically load the scheduler based on its name
        try:
            scheduler_file = Path(find_existing_filename(scheduler_type))
            scheduler_module = importlib.import_module(f"fw.{scheduler_file.stem}")
            scheduler_class = getattr(scheduler_module, scheduler_type)
            scheduler = scheduler_class(self._agent, self._envs)
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
                self.visualize(last_env)
            except AttributeError as e:
                print(e)
        else:
            print(f"No successful training found for environment '{env_type}' to visualize.")

        return metrics


    def visualize(self, env):
        if env is not None:
            gym_env = env.gym_env
            try:
                with self.render(gym_env):
                    self._agent.set_environment(env=gym_env, policy_strategy=PolicyStrategy.REUSE_OTHER_POLICY, other_env_name=gym_env.type_name)
                    self._agent.visualize_trained_model(num_episodes=1, live_animation=self._config["live_animation"])
            except RuntimeError as e:
                print(f"Could not visualize {gym_env.type_name}. {e}")
        else:
            print("No data found to visualize.")
