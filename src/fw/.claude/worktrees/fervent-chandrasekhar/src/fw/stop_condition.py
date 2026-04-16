from enum import auto, Enum
from typing import Optional, Tuple


class Criterion(Enum):
    NBR_POLICY_UPDATES = auto()
    TARGET_REWARD = auto()
    TARGET_REWARD_AND_LOSS = auto()
    AUTO_CONVERGE = auto()


class StopCondition:
    """Defines stopping criteria for training loops.

    Supports multiple stopping strategies such as a fixed number of policy updates
    or achieving a target reward for a number of consecutive evaluations ("patience").

    Attributes:
        stop_criterion (Criterion): The strategy used to determine stopping.
        max_time_steps (int): Maximum allowed training steps.
        target_reward (float, optional): Desired mean reward to reach for stopping.
        target_loss (float, optional): Desired mean loss threshold for stopping.
        patience (int, optional): Number of consecutive evaluations the criteria must be met.
    """

    def __init__(
        self,
        stop_criterion: Criterion = Criterion.NBR_POLICY_UPDATES,
        max_time_steps: int = 2000,
        target_reward: Optional[float]=None,
        target_loss: Optional[float]=None,
        patience: Optional[int] = None
    ):
        """Initializes the StopCondition instance.

        Args:
            stop_criterion (Criterion): Criterion used to decide stopping.
            max_time_steps (int): Maximum number of training steps before stopping.
            target_reward (float, optional): Target reward to reach for stopping.
            target_loss (float, optional): Target loss threshold for stopping.
            patience (int, optional): Number of consecutive successful evaluations needed to stop.
        """
        self.stop_criterion = stop_criterion
        self.max_time_steps = max_time_steps
        self.patience = max(1, patience) if patience is not None else 1
        self._counter = 0

        # Validate criterion-specific parameters
        if stop_criterion == Criterion.TARGET_REWARD:
            if target_reward is None:
                raise ValueError("target_reward must be provided for TARGET_REWARD criterion")
            self.target_reward = target_reward
            self.target_loss = None

        elif stop_criterion == Criterion.TARGET_REWARD_AND_LOSS:
            if target_reward is None or target_loss is None:
                raise ValueError("Both target_reward and target_loss must be provided for TARGET_REWARD_AND_LOSS")
            self.target_reward = target_reward
            self.target_loss = target_loss

        elif stop_criterion == Criterion.AUTO_CONVERGE:
            raise NotImplementedError("AUTO_CONVERGE criterion not yet implemented")

        elif stop_criterion == Criterion.NBR_POLICY_UPDATES:
            self.target_reward = None
            self.target_loss = None


    def should_stop(
        self,
        current_step: int,
        mean_reward: Optional[float] = None,
        mean_loss: Optional[float] = None,
    ) -> Tuple[bool, str]:
        """Evaluates whether the training should stop.

        Args:
            current_step (int): The current number of training steps.
            mean_reward (float, optional): Latest mean reward from evaluation.
            mean_loss (float, optional): Latest mean loss from evaluation.

        Returns:
            Tuple[bool, str]: Whether to stop and the reason.
        """
        if current_step >= self.max_time_steps:
            return True, f"Reached max_time_steps = {self.max_time_steps}"

        if self.stop_criterion == Criterion.TARGET_REWARD:
            if mean_reward is None:
                raise ValueError("mean_reward required for TARGET_REWARD criterion")
            return self._check_condition(
                condition=mean_reward >= self.target_reward,
                success_msg=f"Reached target_reward = {self.target_reward} for {self.patience} consecutive steps"
            )

        if self.stop_criterion == Criterion.TARGET_REWARD_AND_LOSS:
            if mean_reward is None or mean_loss is None:
                raise ValueError("Both mean_reward and mean_loss required for TARGET_REWARD_AND_LOSS")
            return self._check_condition(
                condition=(mean_reward >= self.target_reward and mean_loss <= self.target_loss),
                success_msg=(f"Reached target_reward = {self.target_reward} and "
                             f"target_loss = {self.target_loss} for {self.patience} consecutive steps")
            )

        if self.stop_criterion == Criterion.AUTO_CONVERGE:
            raise NotImplementedError("AUTO_CONVERGE stopping logic is not yet implemented")

        return False, "No stopping condition met"


    def _check_condition(self, condition: bool, success_msg: str) -> Tuple[bool, str]:
        """Helper method to handle counter-based conditions."""
        if condition:
            self._counter += 1
            if self._counter >= self.patience:
                return True, success_msg
        else:
            self._counter = 0
        return False, "Continuing training"


    def __str__(self) -> str:
        """Returns a string representation of the stopping condition.

        Returns:
            str: Description of the stopping condition.
        """
        base = {
            Criterion.NBR_POLICY_UPDATES: f"{self.max_time_steps} policy updates",
            Criterion.TARGET_REWARD: f"target reward {self.target_reward} for {self.patience} consecutive step(s)",
            Criterion.TARGET_REWARD_AND_LOSS: (
                f"target reward {self.target_reward} and target loss {self.target_loss} for {self.patience} consecutive step(s)"
            ),
            Criterion.AUTO_CONVERGE: "automatic convergence detection (not implemented)",
        }.get(self.stop_criterion)

        return base if self.stop_criterion == Criterion.NBR_POLICY_UPDATES else f"{base} (max steps: {self.max_time_steps})"
