import os
import time
import logging

from fw.configuration.config import PolicyStrategy
from fw.scheduling.base_scheduler import BaseScheduler

logger = logging.getLogger(__name__)


class SimpleScheduler(BaseScheduler):
    """
    Scheduler class to manage environment switching and coordinate multi-environment training.
    """

    def __init__(self, agent_mgr, envs):
        """
        Initializes the Scheduler with an agent manager and a list of environments.

        Args:
            agent_mgr: The AgentManager instance responsible for training and policy management.
            envs (list): List of environments used for training.

        Raises:
            RuntimeError: If the provided environment list is empty. Unlike
                `PipelineScheduler`, this scheduler trains over a fixed,
                pre-built list, so a non-empty `envs` is a real precondition
                here - checked in this class rather than in `BaseScheduler`,
                since it isn't a precondition for every scheduler.
        """

        super().__init__(agent_mgr, envs)

        if not self.envs:
            raise RuntimeError("Environments list cannot be empty.")

        self.current_env = None
        self._previous_env_name = None


    @staticmethod
    def _format_reward_weight_suffix(reward_weights) -> str:
        """
        Build a compact filename suffix from reward weights.
        """

        if not reward_weights:
            return "default"

        return "_".join(str(value).replace(".", "") for value in reward_weights.values())


    def _switch_environment(
        self,
        env_index: int,
        policy_strategy: PolicyStrategy = PolicyStrategy.RESET_POLICY,
        other_policy_name: str = None
    ):
        """
        Switches to a different environment.

        Args:
            env_index (int): Index of the environment to switch to.
            policy_strategy (PolicyStrategy, optional): Strategy for handling policies when switching.
            other_policy_name (str, optional): Name of a saved policy to reuse, if applicable.

        Raises:
            RuntimeError: If no environments are available or if the index is invalid.
        """

        if not self.envs:
            raise RuntimeError("No environments available in Scheduler.")

        try:
            self.current_env = self.envs[env_index]
        except IndexError:
            raise RuntimeError(
                f"Invalid environment index '{env_index}'. Only {len(self.envs)} environment(s) available."
            )

        gym_env = self.current_env.gym_env
        sim = getattr(gym_env, "phys_sim", None)

        if sim is not None:
            sim_name = (
                f"{gym_env.dynamics_model.name}-based simulator"
                if hasattr(sim, "_dynamics")
                else None
            )

            if sim_name:
                logger.info("Setting up a %s for a %s ship.", sim_name, gym_env.ship_model.name)

        # Set the desired environment and prepare policy according to the strategy
        self.agent_mgr.set_environment(
            env=self.current_env.gym_env,
            policy_strategy=policy_strategy,
            other_env_name=other_policy_name
        )

        logger.info("[Scheduler] Switched to environment: %s", self.current_env)


    def execute(self, configuration=None):
        """
        Executes the training process across all environments.

        Args:
            configuration (dict, optional): A dictionary where each key is an environment name
                and each value specifies the number of training steps (under "num_steps").

        Returns:
            dict: Metrics for each environment, including training status and duration.

        Raises:
            ValueError: If no environment stop condition can be found in the configuration
                or if the configuration is empty.
        """

        metrics = {}
        policy_strategy = PolicyStrategy.REUSE_OTHER_POLICY

        # Treat a missing configuration the same as an empty one so callers that
        # don't need per-environment stop-condition overrides can omit the argument
        # without hitting a TypeError on the `if env_key in configuration` check.
        if configuration is None:
            configuration = {}

        for i in range(len(self.envs)):
            try:
                # Switch to the next environment
                self._switch_environment(i, policy_strategy, self._previous_env_name)

                logger.info("Training agent '%s' in environment '%s'.", getattr(self.agent_mgr, 'agent', 'Unknown'), self.current_env)

                env_key = str(self.current_env)
                if env_key in configuration:
                    # Look up how many steps to train in this environment
                    stop_condition = configuration[env_key]
                elif len(configuration) == 1:
                    # Single-entry dict whose key no longer matches the active env.
                    original_key = next(iter(configuration))
                    stop_condition = configuration[original_key]
                    logger.info(
                        "[Scheduler] No stop condition for '%s'; "
                        "reusing the one configured for '%s'.", env_key, original_key
                    )
                elif configuration:
                    # Multiple entries but no exact match. Pick the first and warn.
                    original_key = next(iter(configuration))
                    stop_condition = configuration[original_key]
                    logger.warning(
                        "No stop condition found for '%s' (keys: %s). "
                        "Falling back to stop condition for '%s'.",
                        env_key, list(configuration.keys()), original_key
                    )
                else:
                    raise ValueError(
                        f"No stop condition for '{env_key}' and configuration is empty."
                    )

                logger.info("[Scheduler] Training to reach %s in %s.", stop_condition, self.current_env)

                # Reset the patience counter so it doesn't carry over
                # from the previous environment and trigger a premature stop.
                if hasattr(stop_condition, "reset"):
                    stop_condition.reset()

                # Indicate if starting from scratch
                if not self._previous_env_name:
                    logger.info("Starting to train a new %s policy.", self.agent_mgr.agent_type)

                # Train the agent
                start_time = time.time()
                self.agent_mgr.train(stop_condition=stop_condition)
                duration = time.time() - start_time

                # Save policy if reusing it later
                self._previous_env_name = (
                    self.current_env.gym_env.training_name
                    if policy_strategy != PolicyStrategy.RESET_POLICY
                    else None
                )

                if self._previous_env_name:
                    try:
                        env_name = self.current_env.gym_env.training_name
                        save_name = f"{env_name}_{self.agent_mgr.agent_type}_policy"

                        reward_weights = getattr(self.current_env.gym_env, "reward_weights", None)
                        weight_suffix = self._format_reward_weight_suffix(reward_weights)
                        # If reward_weights is empty, use the standard policy name without any suffix.
                        if weight_suffix == "default":
                            self.agent_mgr.agent.save_policy(save_name)
                        else:
                            self.agent_mgr.agent.save_policy(f"{save_name}_{weight_suffix}")

                        # Persist the active profile alongside the checkpoint so
                        # inference / transfer runs can always reconstruct the
                        # exact training configuration without guessing.
                        profile = getattr(self.agent_mgr, "profile", None)
                        if profile is not None:
                            model_dir = getattr(self.agent_mgr.agent, "model_dir", None) or "models"
                            profile_path = os.path.join(model_dir, f"{env_name}_profile.json")
                            try:
                                with open(profile_path, "w", encoding="utf-8") as _fp:
                                    _fp.write(profile.to_json())
                                logger.debug(
                                    "[Scheduler] Profile written to '%s'.", profile_path
                                )
                            except OSError as _pe:
                                logger.warning(
                                    "[Scheduler] Could not save profile JSON (%s).", _pe
                                )
                    except Exception as e:
                        self._previous_env_name = None
                        raise RuntimeError("Failed to save the policy.") from e

                # Log successful training
                metrics[str(self.current_env)] = {"status": "success", "duration": duration}

            except Exception as e:
                logger.exception("Training failed for agent '%s' in '%s': %s", getattr(self.agent_mgr, 'agent', 'Unknown'), self.current_env, e)
                metrics[str(self.current_env)] = {"status": "failure", "error": str(e)}

        logger.info("[Scheduler] Training complete!")

        return metrics
