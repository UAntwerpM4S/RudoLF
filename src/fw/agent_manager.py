import os

from pathlib import Path
from contextlib import contextmanager

from fw.plotly_renderer import PlotlyRenderer
from fw.metrics_computer import MetricsComputer
from fw.episode_collector import EpisodeCollector
from fw.ship_episode_data import CollisionAnnotator
from fw.matplotlib_renderer import MatplotlibRenderer

from fw.environments.base_env import BaseEnv
from fw.environments.lunar_lander import LunarLander
from fw.environments.multi_sim_env import MultiSimEnv
from airsim.communication.airsim_bridge import AirSimBridge

from fw.stop_condition import StopCondition
from fw.policies.base_agent import BaseAgent
from fw.policies.ppo2_agent import PPO2Agent
from fw.policies.multi_ppo_agent import MultiPPOAgent
from fw.policies.single_ppo_agent import SinglePPOAgent
from fw.config import PPO_POLICY_NAME, PPO2_POLICY_NAME, MULTI_PPO_POLICY_NAME, PolicyStrategy, MULTI_POLICY_SINGLE_NAME
from fw.collision_avoidance.scenario_classifier import SCENARIO_LABELS, NUM_SCENARIOS
from fw.policies.multi_policy_single_agent import MultiPolicySingleAgent


class AgentManager:
    """
    AgentManager class for managing agents and environments during training.

    This class includes methods for training, saving/loading policies, and
    visualizing paths taken by the agent during evaluation.
    """

    def __init__(self, agent_type: str, hyperparameters: dict):
        """
        Initialize the AgentManager.

        Args:
            agent_type (str): Type of agent to use (e.g., 'PPO').
            hyperparameters (dict): Hyperparameters for configuring the agent.
        """

        self._agent = None
        self._agent_type = agent_type
        self.prev_smoothing_setting = None
        self._hyperparameters = hyperparameters

    @property
    def agent_type(self) -> str:
        """
        Get the type of agent being used by the agent manager.

        Returns:
            str: The agent type.
        """

        return self._agent_type

    @property
    def agent(self) -> BaseAgent:
        """
        Get the agent that is used by the agent manager.

        Returns:
            BaseAgent: The agent.
        """

        return self._agent

    def _build_single_agent(self, env):
        """
        Create a `SinglePPOAgent` from the given environment.

        If the environment exposes a `sub_environments` attribute, this method
        ensures that exactly one sub-environment is used. Otherwise, the environment
        itself is used directly.

        Args:
            env: Environment instance, optionally containing `sub_environments`.

        Returns:
            SinglePPOAgent: Initialized PPO agent for a single environment.

        Raises:
            RuntimeError: If more than one sub-environment is provided.
        """

        sub_envs = getattr(env, "sub_environments", None)

        if sub_envs is None or len(sub_envs) < 1:
            target_env = env
        elif len(sub_envs) == 1:
            target_env = next(iter(sub_envs.values()))
        else:
            raise RuntimeError("SinglePPOAgent requires exactly one sub-environment.")

        return SinglePPOAgent(environment=target_env, **self._hyperparameters)

    def _build_multi_agent(self, env):
        """
        Create a `MultiPPOAgent` composed of multiple `SinglePPOAgent` instances.

        Each sub-environment in the provided `MultiSimEnv` is assigned its own
        `SinglePPOAgent`. These agents are then combined into a `MultiPPOAgent`
        for coordinated training.

        Args:
            env (MultiSimEnv): Multi-environment containing sub-environments.

        Returns:
            MultiPPOAgent: Initialized multi-agent PPO trainer.

        Raises:
            TypeError: If `env` is not an instance of `MultiSimEnv`.
            ValueError: If no sub-environments are defined.
        """

        if not isinstance(env, MultiSimEnv):
            raise TypeError("MultiPPOAgent requires a MultiSimEnv instance.")

        if not env.sub_environments:
            raise ValueError("MultiSimEnv must contain at least one sub-environment.")

        agents = [
            SinglePPOAgent(environment=sub_env, **self._hyperparameters)
            for sub_env in env.sub_environments.values()
        ]

        return MultiPPOAgent(environment=env, agents=agents, **self._hyperparameters)

    def _build_multi_policy_single_agent(self, env):
        """
        Create a ``MultiPolicySingleAgent`` for the given environment.

        The environment must have ``include_scenario_one_hot=True`` so that
        its observations carry a one-hot suffix the agent uses for policy
        routing.

        Args:
            env: Environment instance (a single PySimColavEnv with one-hot
                encoding enabled, NOT a MultiSimEnv).

        Returns:
            MultiPolicySingleAgent: Initialized multi-policy agent.
        """

        sub_envs = getattr(env, "sub_environments", None)

        if sub_envs is None or len(sub_envs) < 1:
            target_env = env
        elif len(sub_envs) == 1:
            target_env = next(iter(sub_envs.values()))
        else:
            raise RuntimeError(
                "MultiPolicySingleAgent operates on a single environment.  "
                "Got a MultiSimEnv with multiple sub-environments."
            )

        agent = MultiPolicySingleAgent(
            environment=target_env,
            num_policies=NUM_SCENARIOS,
            policy_labels=list(SCENARIO_LABELS),
            **self._hyperparameters,
        )

        # Directory holding the pre-trained per-scenario policies and the
        # spec describing which slots to populate and whether to freeze them.
        # File names are given without the ``.zip`` extension.
        # Slot mapping (from SCENARIO_LABELS):
        #   0 = path-following   ← pretrained_pf
        #   1 = head-on          ← pretrained_ho
        #   2 = overtaking       (no pretrained file, slot stays randomly initialized)
        #   3 = crossing-right   ← pretrained_cr
        # All loaded slots are unfrozen (frozen=False) so they continue to
        # train from the pretrained weights as a warm start.
        pretrained_dir = agent.model_dir
        pretrained_specs = {
            #0: {"file": "pretrained_pf", "frozen": False},
            #1: {"file": "pretrained_ho", "frozen": False},
            #3: {"file": "pretrained_cr", "frozen": False},
        }
        agent.load_pretrained_policies(pretrained_dir, pretrained_specs)

        return agent

    def set_environment(
        self,
        env: BaseEnv,
        policy_strategy: PolicyStrategy = PolicyStrategy.RESET_POLICY,
        other_env_name: str = None,
    ) -> None:
        """
        Assign an environment to the agent and configure the policy strategy.

        Args:
            env (BaseEnv): The environment to set.
            policy_strategy (PolicyStrategy, optional): Strategy for handling
                policy (reset, reuse current, or reuse other).
                Default is PolicyStrategy.RESET_POLICY.
            other_env_name (str, optional): Name of the environment whose policy
                should be reused (if applicable).

        Raises:
            RuntimeError: If an unknown policy type is specified.
        """

        policy_file_name = None

        if policy_strategy == PolicyStrategy.REUSE_CURRENT_POLICY and self._agent:
            policy_file_name = f"{self.get_env().type_name}_{self._agent_type}_policy"
            self._agent.save_policy(policy_file_name)
        elif policy_strategy == PolicyStrategy.REUSE_OTHER_POLICY and other_env_name:
            policy_file_name = f"{other_env_name}_{self._agent_type}_policy"

        if PPO_POLICY_NAME == self._agent_type:
            self._agent = self._build_single_agent(env)
        elif PPO2_POLICY_NAME == self._agent_type:
            self._agent = PPO2Agent(environment=env, **self._hyperparameters)
        elif MULTI_PPO_POLICY_NAME == self._agent_type:
            self._agent = self._build_multi_agent(env)
        elif MULTI_POLICY_SINGLE_NAME == self._agent_type:
            self._agent = self._build_multi_policy_single_agent(env)
        else:
            raise RuntimeError(f"Unknown {self._agent_type} policy type.")

        if policy_file_name:
            full_policy_file_name = f"{policy_file_name}.zip"
            if self.agent.model_dir:
                full_policy_file_name = os.path.join(self.agent.model_dir, full_policy_file_name)

            if Path(full_policy_file_name).exists():
                print(f"Reusing {policy_file_name}.")
                try:
                    self._agent.load_policy(policy_file_name)
                except ValueError as e:
                    print(f"[Fallback] Incompatible model: {e}")
                    print(f"[Fallback] Reverting to RESET_POLICY for {self._agent_type}.")
            else:
                print(f"[Fallback] The policy file {policy_file_name} does not exist.")
                print(f"[Fallback] Reverting to RESET_POLICY for {self._agent_type}.")

    def train(self, stop_condition: StopCondition) -> None:
        """
        Train the agent in the assigned environment.

        Args:
            stop_condition (StopCondition): The condition that determines when
                training should stop.

        Raises:
            RuntimeError: If no agent has been created yet.
        """

        if not self._agent:
            raise RuntimeError("No agent has been created yet!")

        # Start training
        self._agent.learn(stop_condition=stop_condition)

        # Cleanup the environment
        if hasattr(self.get_env(), "fh_sim"):
            self.get_env().fh_sim.dispose()

    @contextmanager
    def smoothing(self, env):
        """
        Context manager that temporarily disables environment smoothing.

        Args:
            env: Environment that may expose an `enable_smoothing` attribute.

        Yields:
            None.
        """

        if hasattr(env, "enable_smoothing"):
            try:
                self.prev_smoothing_setting = env.enable_smoothing
                env.enable_smoothing = False
                yield
            finally:
                env.enable_smoothing = self.prev_smoothing_setting
                self.prev_smoothing_setting = None
        else:
            yield

    @staticmethod
    def _get_ship_env(env, ship_id: str):
        """
        Return the sub-environment for a ship, or the env itself for single-ship.
        """

        if hasattr(env, "sub_environments") and env.sub_environments:
            return env.sub_environments.get(ship_id)
        return env  # single-ship: env IS the ship env

    def visualize_trained_agent(
            self,
            num_episodes: int = 5,
            live_animation: bool = False,
            env_color_index: int = 0,
            airsim_bridge: AirSimBridge = None,
    ) -> None:
        """
        Evaluates and visualizes the trained agent over multiple episodes.

        Args:
            num_episodes: Number of evaluation episodes to run.
            live_animation: If True, renders each episode live.
            env_color_index: Unused; reserved for future color-coding by env.
            airsim_bridge: Optional AirSim bridge for 3-D visualization.

        Raises:
            RuntimeError: If no agent has been created or if the collector
                returns zero steps.
        """

        # Load checkpoint
        # self._agent.policy = PPOPolicy.load("PySimEnv_PPO_policy", device="cpu")

        if not self._agent:
            raise RuntimeError("No agent has been created yet!")

        self.agent.set_policy_eval()
        env = self.get_env()

        with self.smoothing(env):
            raw_env = self.get_env()
            if hasattr(raw_env, "envs"):    # VecEnv wrapping
                raw_env = raw_env.envs[0]
            elif hasattr(raw_env, "env"):   # gym Wrapper
                raw_env = raw_env.env

            # Flip on eval-time realism overrides (variable TS radii, lateral
            # lane spawn, OU wander, late head-on dodge). All overrides are
            # opt-in via env attributes that default to no-op in training.
            if hasattr(raw_env, "set_evaluation_parameters"):
                raw_env.set_evaluation_parameters()
            for sub in getattr(raw_env, "sub_environments", {}).values():
                if hasattr(sub, "set_evaluation_parameters"):
                    sub.set_evaluation_parameters()

            is_lunar_lander = isinstance(raw_env, LunarLander)

            collector = EpisodeCollector(self._agent, raw_env)
            run_data = collector.collect(num_episodes=num_episodes, live_animation=live_animation if not is_lunar_lander else True)

            if not is_lunar_lander:
                # Diagnose empty collection early
                total_steps = sum(
                    sum(len(ep.path) for ep in episodes)
                    for episodes in run_data.ships.values()
                )
                if total_steps == 0:
                    raise RuntimeError(
                        f"EpisodeCollector collected 0 steps across all ships. "
                        f"ship_ids={len(run_data.ships)}, env type={type(raw_env).__name__}. "
                        f"Check that _get_ship_env returns the correct env object."
                    )

                CollisionAnnotator().annotate(run_data)
                metrics = MetricsComputer.compute(run_data)

                for sid, m in metrics.items():
                    ship_env = self._get_ship_env(raw_env, sid)
                    env_name = getattr(ship_env, "env_name", sid)
                    print(f"\n{env_name} evaluation completed!")
                    print(f"Average reward: {m.avg_reward:.2f}")
                    print(f"Best episode reward: {m.best_reward:.2f}")
                    print(f"Worst episode reward: {m.worst_reward:.2f}")
                    print(f"Mean Surge Velocity (mu): {m.mu:.2f}")
                    print(f"Maximum Surge Velocity (Mu): {m.Mu:.2f}")
                    print(f"Mean Track Error Integral (mTEI): {m.mTEI:.2f}")
                    print(f"Maximum Track Error (MTE): {m.MTE:.2f}")
                    print(f"Mean Rudder Angle (mRA): {m.mRA:.3f}")
                    print(f"Maximum Rudder Angle (MRA): {m.MRA:.3f}")
                    print(f"Mean Rudder Angle Change (mRAC): {m.mRAC:.4f}")
                    print(f"Mean LOS Guidance Heading Error (mHE): {m.mHE:.3f} rad")
                    print("Collision rate per policy:")
                    for policy in sorted(m.encounters_per_policy):
                        enc_n = m.encounters_per_policy[policy]
                        col_n = m.collisions_per_policy.get(policy, 0)
                        rate = (col_n / enc_n * 100) if enc_n else 0.0
                        print(f"  {policy:<16s} {rate:6.2f}% ({col_n}/{enc_n} encounters)")
                    n_ep = m.num_episodes
                    out_rate = (m.collisions_outside_encounters / n_ep * 100) if n_ep else 0.0
                    print(f"  outside-encounter {out_rate:6.2f}% ({m.collisions_outside_encounters}/{n_ep} episodes)")
                    in_enc = m.collision_count - m.collisions_outside_encounters
                    total_denom = m.num_encounters + n_ep
                    cr_pct = (m.collision_count / total_denom * 100) if total_denom else 0.0
                    print(
                        f"Total collision rate (CR): {cr_pct:.2f}% "
                        f"({m.collision_count} = {in_enc} in {m.num_encounters} encounters "
                        f"+ {m.collisions_outside_encounters} outside / {n_ep} episodes)"
                    )
                    print(f"Collision per type: {m.collision_type_count}")
                    print(f"Collision per side: {m.collision_side_count}")
                    print(f"COLREG violation count: {m.colreg_violation_count}")

                    # Scenario classification diagnostics — per encounter
                    # (one record captured at encounter entry).
                    enc_rate = (
                        m.encounter_misclassification_count / m.num_encounters * 100
                        if m.num_encounters else 0.0
                    )
                    print(
                        f"Scenario misclassification — "
                        f"{m.encounter_misclassification_count}/{m.num_encounters} encounters "
                        f"({enc_rate:.2f}%)"
                    )
                    if m.encounter_confusion:
                        print("Encounter confusion (gt → pred: count):")
                        for (gt, pred), cnt in sorted(m.encounter_confusion.items()):
                            tag = "" if gt == pred else "  [MISMATCH]"
                            print(f"  {gt} → {pred}: {cnt}{tag}")

                save_dir = self.agent.model_dir or "."

                for sid in metrics.keys():
                    ship_env = self._get_ship_env(raw_env, sid)
                    env_name = getattr(ship_env, "env_name", sid)
                    if hasattr(ship_env, "plot_min_distance_stats"):
                        ship_env.plot_min_distance_stats(
                            pdf_save_path=f"{save_dir}/{env_name}_min_distance_PDF.png",
                            boxplot_save_path=f"{save_dir}/{env_name}_min_distance_boxplots.png",
                        )
                    if hasattr(ship_env, "plot_scenario_confusion_matrix"):
                        ship_env.plot_scenario_confusion_matrix(
                            save_path=f"{save_dir}/{env_name}_scenario_confusion_matrix.png",
                        )

                MatplotlibRenderer.render(
                    run_data.ships, metrics, raw_env,
                    save_path=f"{save_dir}/trajectory.png",
                )

                # PlotlyRenderer.render(
                #     run_data.ships, metrics,
                # )

    def get_env(self):
        """
        Retrieve the environment associated with the agent.

        Returns:
            The environment object.
        """

        return self._agent.get_env()
