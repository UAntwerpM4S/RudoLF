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
            RuntimeError: If `envs` is empty (checked here, not in BaseScheduler,
                because PipelineScheduler does not need a pre-built list).
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
        Switch to the environment at `env_index` and prepare the policy.

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

        self.agent_mgr.set_environment(
            env=self.current_env.gym_env,
            policy_strategy=policy_strategy,
            other_env_name=other_policy_name
        )

        logger.info("[Scheduler] Switched to environment: %s", self.current_env)


    def execute(self, configuration=None):
        """
        Train the agent on each environment in turn, reusing the previous policy.

        Args:
            configuration (dict, optional): Maps environment name (str(env)) to its
                stop condition. With no exact match, the first entry is reused.

        Returns:
            dict: Per environment, {"status": "success", "duration"} or
                {"status": "failure", "error"}. Errors per environment (e.g. an
                empty configuration, a failed save) are logged and recorded, not raised.
        """

        metrics = {}
        policy_strategy = PolicyStrategy.REUSE_OTHER_POLICY

        if configuration is None:
            configuration = {}

        for i in range(len(self.envs)):
            try:
                self._switch_environment(i, policy_strategy, self._previous_env_name)

                logger.info("Training agent '%s' in environment '%s'.", getattr(self.agent_mgr, 'agent', 'Unknown'), self.current_env)

                env_key = str(self.current_env)
                if env_key in configuration:
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

                # Reset patience so it does not carry over from the previous env.
                if hasattr(stop_condition, "reset"):
                    stop_condition.reset()

                if not self._previous_env_name:
                    logger.info("Starting to train a new %s policy.", self.agent_mgr.agent_type)

                start_time = time.time()
                self.agent_mgr.train(stop_condition=stop_condition)
                duration = time.time() - start_time

                # Save the policy so the next environment can reuse it
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
                        # The canonical name is what AgentManager.set_environment,
                        # Trainer.visualize and the next env load. Envs with reward
                        # weights also get a weight-suffixed copy for bookkeeping.
                        self.agent_mgr.agent.save_policy(save_name)
                        if weight_suffix != "default":
                            self.agent_mgr.agent.save_policy(f"{save_name}_{weight_suffix}")

                        # Save the active profile next to the checkpoint for
                        # inference / transfer runs.
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

                metrics[str(self.current_env)] = {"status": "success", "duration": duration}

            except Exception as e:
                logger.exception("Training failed for agent '%s' in '%s': %s", getattr(self.agent_mgr, 'agent', 'Unknown'), self.current_env, e)
                metrics[str(self.current_env)] = {"status": "failure", "error": str(e)}

        logger.info("[Scheduler] Training complete!")

        return metrics
