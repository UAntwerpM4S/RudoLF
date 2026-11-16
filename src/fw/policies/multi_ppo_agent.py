import os
import copy
import torch
import zipfile
import tempfile
import logging
import numpy as np

from fw.policies.ppo_policy import PPOPolicy, apply_policy_yaw_rate_obs_scale, require_policy_dims
from fw.policies.base_ppo_agent import BasePPOAgent
from fw.policies.single_ppo_agent import SinglePPOAgent
from fw.configuration.stop_condition import StopCondition
from fw.common.tools.logging_utils import dump_multi_agent_metrics
from fw.configuration.profile_applier import SimulatorApplier
from fw.configuration.config_profile import VesselProfile
from fw.environments.multi_sim_env import MultiSimEnv
from fw.common.tools.ppo_utils import reseed_env_rng
from typing import Optional, Tuple, Union, Any

logger = logging.getLogger(__name__)


class MultiPPOAgent(BasePPOAgent):
    """
    Multi-agent PPO trainer for a `MultiSimEnv`.

    Coordinates independent `SinglePPOAgent` instances, one per sub-environment.
    Rollouts are collected synchronously; updates are per agent. Fully
    decentralized: own observations, no shared policies, no central critic.

    Attributes:
        agents (list[SinglePPOAgent]): List of PPO agents (one per environment).
        num_envs (int): Number of sub-environments.
        max_nbr_iterations (int): Maximum rollout length.
        normalize (bool): Whether to normalize rewards per trajectory.
    """

    MINIMUM_ITERATIONS_FOR_BEST_SAVE = 20

    def __init__(
        self,
        profile: VesselProfile,
        *args,
        agents: list[SinglePPOAgent],
        **kwargs
    ):
        """
        Initialize the multi-agent PPO trainer.

        Args:
            profile: Profile that will be used for training or visualization.
            *args: Positional arguments forwarded to `BasePPOAgent`.
            agents (list[SinglePPOAgent]): List of initialized agents, one per sub-environment.
            **kwargs: Keyword arguments forwarded to `BasePPOAgent`.

        Raises:
            TypeError: If the environment is not a `MultiSimEnv`.
            ValueError: If the number of agents does not match the number of sub-environments.
        """

        super().__init__(profile, *args, **kwargs)

        self.agents = agents
        self.num_envs = len(self.env.sub_environments)

        if not isinstance(self.env, MultiSimEnv):
            raise TypeError("MultiPPOAgent requires MultiSimEnv")

        if len(agents) != self.num_envs:
            raise ValueError(
                f"Number of agents ({len(agents)}) must match "
                f"number of sub-environments ({self.num_envs})"
            )

        # Sub-environments are stepped directly (no worker copies), so seed them
        # here: one child seed per sub-environment, in order.
        for sub_env, sub_seed in zip(self.env.sub_environments.values(),
                                     self._spawn_worker_seeds(self.num_envs)):
            reseed_env_rng(sub_env, sub_seed)


    def predict(
            self,
            state: Union[np.ndarray, Tuple[np.ndarray, ...]],
            deterministic: bool = False
    ) -> Tuple[Union[np.ndarray, Tuple[np.ndarray, ...]], np.ndarray, np.ndarray]:
        """
        Compute actions and value estimates for all agents.

        Args:
            state (Union[np.ndarray, Tuple[np.ndarray, ...]]): Tuple of observations,
                one per agent/environment.
            deterministic (bool): Whether to use deterministic policy outputs.

        Returns:
            Tuple (actions per agent, log-probabilities (num_envs,), values
            (num_envs,)), in agent order.
        """

        actions = []
        log_probs = []
        values = []

        for i, agent in enumerate(self.agents):
            action, log_prob, value = agent.predict(
                state[i],
                deterministic=deterministic
            )

            actions.append(action)
            log_probs.append(log_prob)
            values.append(value)

        return (
            tuple(actions),
            np.stack(log_probs),
            np.stack(values),
        )


    def _reset_envs(self, num_envs):
        """
        Reset the multi-environment.

        Args:
            num_envs (int): Number of environments (unused, inferred from env).

        Returns:
            Initial observations, one per sub-environment.
        """

        for sub_env in self.env.sub_environments.values():
            if hasattr(sub_env, "randomize"):
                sub_env.randomize()

            # Start every episode from the profile's pose, not the env default.
            if self._profile is not None:
                _sim = getattr(sub_env, "phys_sim", None) or getattr(sub_env, "_phys_sim", None)
                if _sim is not None:
                    SimulatorApplier.apply(self._profile, _sim)

        states, _ = self.env.reset()
        return states


    def _step_envs(self, actions):
        """
        Step all sub-environments with joint actions.

        Args:
            actions (tuple): Tuple of actions, one per agent.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
                - next_states: Next observations (auto-reset ones for finished envs).
                - rewards: Rewards per agent.
                - terminations: Termination flags.
                - truncations: Truncation flags.
                - final_states: Real last observations (used to bootstrap truncations).

        Rewards and flags are taken from the per-env info dicts when all of
        them provide "rewards", "terminations" and "truncations"; otherwise the
        standard step outputs are used.
        """

        infos: list[dict[str, Any]]
        next_states, next_rewards, next_terminations, truncations, infos = self.env.step(actions)

        if infos and all(
                all(k in env_info for k in ("rewards", "terminations", "truncations"))
                for env_info in infos
        ):
            rewards = np.array([env_info["rewards"] for env_info in infos], dtype=np.float32)
            terminations = np.array([env_info["terminations"] for env_info in infos], dtype=bool)
            truncations = np.array([env_info["truncations"] for env_info in infos], dtype=bool)
        else:
            rewards = next_rewards
            terminations = next_terminations

        # MultiSimEnv does not auto-reset, so the returned observations are
        # already the real last observations of finished sub-environments.
        return next_states, rewards, terminations, truncations, next_states


    def _predict(self, states, deterministic=False):
        """
        Per-agent prediction for the rollout, including the raw pre-tanh actions.

        Args:
            states (np.ndarray or tuple): Observations per agent.
            deterministic (bool): Whether to use deterministic actions.

        Returns:
            Tuple (actions per agent, raw actions per agent, log-probabilities
            (num_envs,), values (num_envs,)), in agent order.
        """

        actions = []
        raw_actions = []
        log_probs = []
        values = []

        for i, agent in enumerate(self.agents):
            action, raw_action, log_prob, value = agent.policy.predict_with_raw(
                states[i],
                deterministic=deterministic,
            )

            actions.append(action[0])
            raw_actions.append(raw_action[0])
            log_probs.append(log_prob.item())
            values.append(value.item())

        return (
            tuple(actions),
            tuple(raw_actions),
            np.stack(log_probs),
            np.stack(values),
        )


    def _get_bootstrap_value(self, next_state, env_id: Optional[int] = None):
        """
        Estimate value for bootstrapping for a specific agent.

        Args:
            next_state (np.ndarray): State to evaluate.
            env_id (Optional[int]): Index of the agent/environment.

        Returns:
            float: Estimated value.

        Raises:
            TypeError: If `env_id` is None.
        """

        if env_id is None:
            raise TypeError(
                "_get_bootstrap_value() requires an integer env_id; got None. "
                "Pass the index of the agent whose value function should be queried."
            )

        with torch.no_grad():
            last_state = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
            _, _, next_values = self.agents[env_id].policy.network(last_state)
            return next_values.squeeze(0).item()


    def learn(
            self,
            stop_condition: Optional[StopCondition] = None,
    ) -> None:
        """
        Train all agents using parallel rollouts.

        Args:
            stop_condition (Optional[StopCondition]): Stopping criterion. If None,
                a default condition is used.

        Each iteration collects one trajectory per sub-environment and trains
        each agent on its own (matched by env_id tag). A missing trajectory is
        skipped with a warning and logged as NaN. Saves best policies per agent,
        periodic checkpoints and per-agent metrics.

        Early stopping is per agent: each has its own copy of the stop
        condition, judged on its own reward. An agent that meets it is frozen
        (no more updates, still acting and logged with frozen=True). Training
        ends when all agents are frozen or after `max_time_steps` iterations.
        """

        logger.info("Starting training with %d parallel environments on a '%s' device", self.num_envs, self.device)

        last_save_index = 0
        best_rewards = [float('-inf')] * len(self.agents)
        training_metrics = {f"agent{i}": [] for i in range(len(self.agents))}
        stopping_condition = stop_condition if stop_condition is not None else StopCondition()

        # StopCondition is stateful (patience counter), hence one copy per agent.
        agent_stop_conditions = [copy.deepcopy(stopping_condition) for _ in self.agents]
        for condition in agent_stop_conditions:
            condition.reset()
        frozen = [False] * len(self.agents)

        best_policy_files = [
            f"Best_{env.training_name}_{i+1}_PPO_policy" if self.num_envs > 1
            else f"Best_{env.training_name}_PPO_policy"
            for i, env in enumerate(self.env.sub_environments.values())
        ]

        try:
            for iteration in range(stopping_condition.max_time_steps):
                self.set_policy_eval()

                batch_data = self.run_parallel_episodes(self.num_envs)

                # Match episodes to agents by env_id.
                episode_by_env: dict[int, dict] = {
                    ep["env_id"]: ep
                    for ep in batch_data
                    if "env_id" in ep
                }

                # Fallback without env_id tags: match by position.
                if not episode_by_env:
                    episode_by_env = {i: ep for i, ep in enumerate(batch_data)}

                # Train each agent that has not met its own stop condition yet
                for agent_id, agent in enumerate(self.agents):
                    episode = episode_by_env.get(agent_id)

                    if episode is None:
                        logger.warning("[Warning] Iteration %d: no trajectory for agent %s - skipping update.", iteration, agent_id)
                        if not frozen[agent_id]:
                            # A gap breaks the patience streak.
                            agent_stop_conditions[agent_id].reset()
                        training_metrics[f"agent{agent_id}"].append({
                            'iteration': iteration,
                            'loss': float('nan'),
                            'advantage': float('nan'),
                            'reward': float('nan'),
                            'frozen': frozen[agent_id],
                        })
                        continue

                    mean_reward = np.mean(episode['rewards'])
                    mean_advantage = np.mean(episode['advantages'])

                    if frozen[agent_id]:
                        # Converged: no update, reward still logged.
                        training_metrics[f"agent{agent_id}"].append({
                            'iteration': iteration,
                            'loss': float('nan'),
                            'advantage': mean_advantage,
                            'reward': mean_reward,
                            'frozen': True,
                        })
                        continue

                    loss = agent.train([episode])

                    training_metrics[f"agent{agent_id}"].append({
                        'iteration': iteration,
                        'loss': loss,
                        'advantage': mean_advantage,
                        'reward': mean_reward,
                        'frozen': False,
                    })

                    if iteration >= self.MINIMUM_ITERATIONS_FOR_BEST_SAVE and mean_reward > best_rewards[agent_id]:
                        best_rewards[agent_id] = mean_reward
                        agent.save_policy(best_policy_files[agent_id])

                    if iteration % self.eval_frequency == 0:
                        logger.info("Iteration %d | agent %s | average reward: %.3f, loss: %.3f", iteration, agent_id, mean_reward, loss)

                    # Stop decision on this agent's own reward.
                    agent_done, reason = agent_stop_conditions[agent_id].should_stop(
                        current_step=iteration,
                        mean_reward=mean_reward,
                        mean_loss=loss,
                    )
                    if agent_done:
                        frozen[agent_id] = True
                        logger.info("Iteration %d | agent %s frozen: %s", iteration, agent_id, reason)
                        # Frozen before MINIMUM_ITERATIONS_FOR_BEST_SAVE: save
                        # the converged policy so a Best_* file exists.
                        if best_rewards[agent_id] == float('-inf'):
                            best_rewards[agent_id] = mean_reward
                            agent.save_policy(best_policy_files[agent_id])

                # Periodic checkpoint (all agents in one zip)
                if iteration > 0 and iteration % self.eval_frequency == 0:
                    self.save_policy(f"ppo_checkpoint_{iteration}")

                if all(frozen):
                    logger.info("Stopping: every agent met its stop condition (iteration %d).", iteration)
                    last_save_index = iteration
                    break

            if not all(frozen):
                logger.info(
                    "Stopping: reached max_time_steps = %d; agent(s) %s did not meet their stop "
                    "condition (frozen: %s).",
                    stopping_condition.max_time_steps,
                    [i for i, f in enumerate(frozen) if not f],
                    [i for i, f in enumerate(frozen) if f] or "none",
                )

            # Final checkpoint and metrics only after normal completion.
            self.save_policy(f'ppo_checkpoint_{last_save_index if last_save_index > 0 else stopping_condition.max_time_steps}')
            dump_multi_agent_metrics(self.model_dir, training_metrics)
        finally:
            # Always close the sub-environments (simulator connections).
            for sub_env in self.env.sub_environments.values():
                try:
                    sub_env.close()
                except Exception:
                    logger.exception("Closing sub-environment %s failed.", getattr(sub_env, "training_name", sub_env))

        logger.info("Training completed!")


    def load_policy(self, policy_file_name: str) -> None:
        """
        Load a multi-agent policy from a ZIP archive.

        Args:
            policy_file_name (str): Path to the ZIP archive containing agent policies.

        The archive holds one serialized policy per agent: policy_0.pth,
        policy_1.pth, ...

        Each agent's sub-environment takes over the yaw-rate observation scale its
        policy was trained with (apply_policy_yaw_rate_obs_scale).

        Raises:
            FileNotFoundError: If the archive does not exist.
            RuntimeError: If loading fails or required files are missing.
            ValueError: If a policy's observation/action size differs from its
                agent's, or its stored yaw-rate scale is invalid; no agent is
                changed then.
        """

        if not policy_file_name.endswith(".zip"):
            policy_file_name += ".zip"

        if self.model_dir:
            policy_file_name = os.path.join(self.model_dir, policy_file_name)

        if not os.path.exists(policy_file_name):
            raise FileNotFoundError(f"Policy archive not found: {policy_file_name}")

        with tempfile.TemporaryDirectory() as tmp_dir:
            try:
                with zipfile.ZipFile(policy_file_name, 'r') as main_zip:
                    file_list = main_zip.namelist()
                    expected_files = {f"policy_{i}.pth" for i in range(len(self.agents))}
                    if not expected_files.issubset(set(file_list)):
                        missing = expected_files - set(file_list)
                        raise RuntimeError(f"Archive missing required files: {missing}")

                    main_zip.extractall(tmp_dir)

                # Load and check every policy before assigning any, so a failure
                # leaves all agents as they were.
                policies = []
                for i, agent in enumerate(self.agents):
                    agent_path = os.path.join(tmp_dir, f"policy_{i}.pth")

                    if not os.path.exists(agent_path):
                        raise FileNotFoundError(f"Missing policy file: policy_{i}.pth")

                    saved_variables = torch.load(agent_path, map_location=self.device, weights_only=False)

                    if agent.policy is None:
                        raise RuntimeError(f"Agent {i} has no initialized policy.")

                    policy = PPOPolicy.build_policy_from_state(saved_variables, self.device)
                    require_policy_dims(policy, agent.policy.input_dim, agent.policy.output_dim,
                                        f"{policy_file_name}:policy_{i}.pth")
                    policies.append(policy)

                for i, (agent, policy) in enumerate(zip(self.agents, policies)):
                    # Each agent's sub-environment takes over its policy's yaw-rate scale.
                    apply_policy_yaw_rate_obs_scale(policy, agent.env, f"{policy_file_name}:policy_{i}.pth")
                    agent.policy = policy
                    # Rebind the optimizer to the loaded network; otherwise train()
                    # would update the old, discarded parameters.
                    agent.optimizer = torch.optim.Adam(
                        agent.policy.network.parameters(), lr=agent.learning_rate
                    )
                    # Restoring Adam momentum is deliberately disabled (not needed to
                    # resume training), as in SinglePPOAgent.
                    # if "optimizer_state_dict" in saved_variables:
                    #     try:
                    #         agent.optimizer.load_state_dict(saved_variables["optimizer_state_dict"])
                    #     except Exception as opt_exc:  # shape mismatch on arch change, not fatal
                    #         print("Could not restore optimizer state for agent %d: %s", i, opt_exc)

                if any(agent.policy is None for agent in self.agents):
                    raise RuntimeError("Some agents failed to load policies")

            except ValueError:
                raise   # size or stored-scale check above
            except Exception as e:
                raise RuntimeError(f"Failed to load multi-agent policy: {e}") from e


    def save_policy(self, policy_file_name: str) -> None:
        """
        Save all agent policies into a single ZIP archive.

        Args:
            policy_file_name (str): Output file name ('.zip' will be appended if missing).

        Each agent is stored as policy_<i>.pth (policy.serialize_policy()).

        Raises:
            RuntimeError: If saving fails (including a missing policy file).
        """

        if not policy_file_name.endswith(".zip"):
            policy_file_name += ".zip"

        full_path = os.path.join(self.model_dir, policy_file_name) if self.model_dir else policy_file_name

        with tempfile.TemporaryDirectory() as tmp_dir:
            try:
                # Policy only; saving the optimizer state is disabled.
                for i, agent in enumerate(self.agents):
                    agent_path = os.path.join(tmp_dir, f"policy_{i}.pth")

                    if agent.policy is None:
                        raise RuntimeError(f"Agent {i} has no policy to save.")

                    checkpoint = agent.policy.serialize_policy()
                    # if hasattr(agent, "optimizer") and agent.optimizer is not None:
                    #     checkpoint["optimizer_state_dict"] = agent.optimizer.state_dict()

                    torch.save(
                        checkpoint,
                        agent_path,
                    )

                with zipfile.ZipFile(full_path, 'w', compression=zipfile.ZIP_DEFLATED) as main_zip:
                    for i in range(len(self.agents)):
                        file_name = f"policy_{i}.pth"
                        file_path = os.path.join(tmp_dir, file_name)

                        if not os.path.exists(file_path):
                            raise FileNotFoundError(f"Expected policy file {file_name} not found.")

                        main_zip.write(file_path, arcname=file_name)

            except Exception as e:
                raise RuntimeError(f"Failed to save multi-agent policy to '{full_path}'.") from e


    def set_policy_eval(self):
        """
        Set all agent policy networks to evaluation mode.
        """

        for agent in self.agents:
            agent.policy.network.eval()
