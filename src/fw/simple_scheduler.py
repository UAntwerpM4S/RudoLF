import time
import logging
import traceback

from fw.config import PolicyStrategy
from fw.base_scheduler import BaseScheduler


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
        """

        super().__init__(agent_mgr, envs)

        self.current_env = None
        self._previous_env_name = None


    @staticmethod
    def _format_reward_weight_suffix(reward_weights) -> str:
        """Build a compact filename suffix from reward weights."""
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
                else f"{getattr(sim, '_engine')} simulator"
                if hasattr(sim, "_engine")
                else None
            )

            if sim_name:
                print(f"Setting up a {sim_name} for a {gym_env.ship_model.name} ship.")

        # Set the desired environment and prepare policy according to the strategy
        self.agent_mgr.set_environment(
            env=self.current_env.gym_env,
            policy_strategy=policy_strategy,
            other_env_name=other_policy_name
        )

        print(f"[Scheduler] Switched to environment: {self.current_env}")


    def execute(self, configuration=None):
        """
        Executes the training process across all environments.

        Args:
            configuration (dict, optional): A dictionary where each key is an environment name
                and each value specifies the number of training steps (under "num_steps").

        Returns:
            dict: Metrics for each environment, including training status and duration.
        """

        metrics = {}
        policy_strategy = PolicyStrategy.REUSE_OTHER_POLICY

        for i in range(len(self.envs)):
            try:
                # Switch to the next environment
                self._switch_environment(i, policy_strategy, self._previous_env_name)

                logging.info(f"Training agent '{getattr(self.agent_mgr, 'agent', 'Unknown')}' in environment '{self.current_env}'.")

                # Look up how many steps to train in this environment
                stop_condition = configuration[str(self.current_env)]
                print(f"[Scheduler] Training to reach {stop_condition} in {self.current_env}.")

                # Indicate if starting from scratch
                if not self._previous_env_name:
                    print(f"Starting to train a new {self.agent_mgr.agent_type} policy.")

                # Train the agent
                start_time = time.time()
                self.agent_mgr.train(stop_condition=stop_condition)
                duration = time.time() - start_time

                # Save policy if reusing it later
                self._previous_env_name = self.current_env if policy_strategy != PolicyStrategy.RESET_POLICY else None

                if self._previous_env_name:
                    try:
                        save_name = f"{self.current_env}_{self.agent_mgr.agent_type}_policy"
                        self.agent_mgr.agent.save_policy(save_name)

                        reward_weights = getattr(self.current_env.gym_env, "reward_weights", None)
                        weight_suffix = self._format_reward_weight_suffix(reward_weights)
                        self.agent_mgr.agent.save_policy(f"{save_name}_{weight_suffix}")
                    except Exception as e:
                        self._previous_env_name = None
                        raise RuntimeError("Failed to save the policy.") from e

                # Log successful training
                metrics[str(self.current_env)] = {"status": "success", "duration": duration}

            except Exception as e:
                traceback.print_exc()
                logging.error(f"Training failed for agent '{getattr(self.agent_mgr, 'agent', 'Unknown')}' in '{self.current_env}': {e}")
                metrics[str(self.current_env)] = {"status": "failure", "error": str(e)}

        print("[Scheduler] Training complete!")

        return metrics
