class BaseScheduler:
    """
    Base scheduler for managing environment switching and coordinating multi-environment training.
    """

    def __init__(self, agent_mgr, envs=None):
        """
        Initializes the BaseScheduler.

        Args:
            agent_mgr: The agent manager instance responsible for training.
            envs: Optional list of training environments; empty for schedulers
                that build their environments lazily.
        """

        self.agent_mgr = agent_mgr
        self.envs = envs if envs is not None else []

    def execute(self, configuration=None):
        """
        Run the schedule, e.g. switching environments on step counts,
        performance thresholds or a curriculum.

        Args:
            configuration: Optional training configuration (step limits,
                switching conditions, curriculum parameters, ...).

        Raises:
            NotImplementedError: Must be overridden in a subclass.
        """

        # Example logic in a subclass:
        # for env in self.envs:
        #     self.agent_mgr.train(env, configuration)
        #     if agent_mgr.performance(env) > some_threshold:
        #         move to next environment

        raise NotImplementedError
