class BaseScheduler:
    """
    Base scheduler for managing environment switching and coordinating multi-environment training.
    """

    def __init__(self, agent_mgr, envs=None):
        """
        Initializes the BaseScheduler.

        Args:
            agent_mgr: The agent manager instance responsible for training.
            envs: Optional list of environments to be used during training.
                Defaults to an empty list for schedulers that build their
                environments lazily instead of receiving them up front.
        """

        self.agent_mgr = agent_mgr
        self.envs = envs if envs is not None else []

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
        #     self.agent_mgr.train(env, configuration)
        #     if agent_mgr.performance(env) > some_threshold:
        #         move to next environment

        raise NotImplementedError
