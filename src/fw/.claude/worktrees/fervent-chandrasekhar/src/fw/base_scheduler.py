class BaseScheduler:
    """Base scheduler for managing environment switching and coordinating multi-environment training."""

    def __init__(self, agent, envs):
        """
        Initializes the BaseScheduler.

        Args:
            agent: The agent instance responsible for training.
            envs: List of environments to be used during training.

        Raises:
            RuntimeError: If the provided environment list is empty.
        """
        if not envs:
            raise RuntimeError("Environments list cannot be empty.")

        self.envs = envs
        self.agent = agent


    def execute(self, configuration=None):
        """
        Executes the scheduling process.

        This method coordinates training across multiple environments. A typical implementation
        could include switching environments based on the number of steps, performance thresholds,
        curriculum strategies, or a predefined schedule.

        Args:
            configuration: Optional configuration specifying how the agent should train
                in a given environment. May include step limits, switching conditions,
                or curriculum parameters.

        Raises:
            NotImplementedError: Always, since this method must be overridden in a subclass.
        """
        # Example logic in a subclass:
        # for env in self.envs:
        #     self.agent.train(env, configuration)
        #     if agent.performance(env) > some_threshold:
        #         move to next environment
        raise NotImplementedError
