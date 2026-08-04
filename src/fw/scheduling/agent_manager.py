import os
import math

from pathlib import Path
from fw.configuration.stop_condition import StopCondition
from fw.configuration.config_profile import VesselProfile
from fw.configuration.profile_applier import apply_profile
from fw.postprocessing.metrics_computer import MetricsComputer
from fw.postprocessing.episode_collector import EpisodeCollector
from fw.postprocessing.ship_episode_data import CollisionAnnotator
from fw.postprocessing.matplotlib_renderer import MatplotlibRenderer

from fw.environments.base_env import BaseEnv
from fw.environments.lunar_lander import LunarLander
from fw.environments.multi_sim_env import MultiSimEnv
from airsim.communication.airsim_bridge import AirSimBridge

from fw.policies.base_agent import BaseAgent
from fw.policies.ppo2_agent import PPO2Agent
from fw.policies.multi_ppo_agent import MultiPPOAgent
from fw.policies.single_ppo_agent import SinglePPOAgent
from fw.policies.multi_policy_single_agent import MultiPolicySingleAgent
from fw.collision_avoidance.scenario_classifier import SCENARIO_LABELS, NUM_SCENARIOS
from fw.configuration.config import PPO_POLICY_NAME, PPO2_POLICY_NAME, MULTI_PPO_POLICY_NAME, PolicyStrategy, MULTI_POLICY_SINGLE_NAME


class AgentManager:
    """
    AgentManager class for managing agents and environments during training.

    This class includes methods for training, saving/loading policies, and
    visualizing paths taken by the agent during evaluation.
    """

    def __init__(self, agent_type: str, profile: VesselProfile, hyperparameters: dict):
        """
        Initialize the AgentManager.

        Args:
            agent_type (str): Type of agent to use (e.g., 'PPO').
            profile (VesselProfile): Profile that will be used for training or visualization.
            hyperparameters (dict): Hyperparameters for configuring the agent.
        """

        self._agent = None
        self._profile = profile
        self._agent_type = agent_type
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
    def profile(self) -> "VesselProfile":
        """
        Get the active VesselProfile held by this manager.

        Returns:
            VesselProfile: The profile used for training or visualization.
        """

        return self._profile

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

        return SinglePPOAgent(self._profile, environment=target_env, **self._hyperparameters)

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
            SinglePPOAgent(self._profile, environment=sub_env, **self._hyperparameters)
            for sub_env in env.sub_environments.values()
        ]

        return MultiPPOAgent(self.profile, environment=env, agents=agents, **self._hyperparameters)

    def _build_multi_policy_single_agent(self, env):
        """
        Create a `MultiPolicySingleAgent` for the given environment.

        The environment must have `include_scenario_one_hot=True` so that
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
        # File names are given without the `.zip` extension.
        # Slot mapping (from SCENARIO_LABELS):
        #   0 = path-following   <- pretrained_pf
        #   1 = head-on          <- pretrained_ho
        #   2 = overtaking       (no pretrained file, slot stays randomly initialized)
        #   3 = crossing-right   <- pretrained_cr
        # All loaded slots are unfrozen (frozen=False) so they continue to
        # train from the pretrained weights as a warm start.
        pretrained_dir = agent.model_dir
        pretrained_specs = {
            #0: {"file": "pretrained_pf", "frozen": False},
            #1: {"file": "pretrained_ho", "frozen": False},
            #2: {"file": "pretrained_ot", "frozen": True},
            #3: {"file": "pretrained_cr", "frozen": True},
        }
        agent.load_pretrained_policies(pretrained_dir, pretrained_specs)

        return agent

    def set_environment(
        self,
        env: BaseEnv,
        policy_strategy: PolicyStrategy = PolicyStrategy.RESET_POLICY,
        other_env_name: str = None,
        profile: VesselProfile = None,
        hyperparameters: dict = None,
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
            profile (VesselProfile, optional): Replace the manager's active
                profile before building the new agent. Defaults to None,
                which keeps whichever profile the manager already holds -
                this preserves the exact behavior every existing caller
                (Trainer, SimpleScheduler) relies on. `fw.pipeline_scheduler`
                passes a different profile here per pipeline block, so one
                AgentManager can be reused across a whole JSON-driven run
                instead of constructing a new one per block.
            hyperparameters (dict, optional): Replace the manager's active
                hyperparameters before building the new agent. Defaults to
                None, which keeps the hyperparameters already held. Same
                rationale as `profile` above.

        Raises:
            RuntimeError: If an unknown policy type is specified.
        """

        if profile is not None:
            self._profile = profile

        if hyperparameters is not None:
            self._hyperparameters = hyperparameters

        policy_file_name = None

        if policy_strategy == PolicyStrategy.REUSE_CURRENT_POLICY and self._agent:
            policy_file_name = f"{self.get_env().training_name}_{self._agent_type}_policy"
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

        # Always propagate profile env settings (e.g. enable_smoothing) to the
        # environment, regardless of which entry point called set_environment.
        apply_profile(self._profile, env=env)

        if policy_file_name:
            full_policy_file_name = f"{policy_file_name}.zip"
            if self.agent.model_dir:
                full_policy_file_name = os.path.join(self.agent.model_dir, full_policy_file_name)

            if Path(full_policy_file_name).exists():
                print(f"Reusing {policy_file_name}.")
                try:
                    self._agent.load_policy(policy_file_name)
                except (ValueError, RuntimeError) as e:
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
        fh_sim = getattr(self.get_env(), "fh_sim", None)
        if fh_sim is not None:
            fh_sim.dispose()

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
            return_summary: bool = False,
            show_plots: bool = True
        ) -> dict | None:

        """
        Evaluates and visualizes the trained agent over multiple episodes.

        Args:
            num_episodes (int, optional): Number of episodes to visualize (default is 5).
            live_animation (bool, optional): Enable live animation of the agent in the environment.
            env_color_index (int, optional): Index of the environment color.
            airsim_bridge (AirSimBridge, optional): AirSimBridge to create a connection with the Cosys AirSim.
            return_summary (bool, optional): Return evaluation summary metrics when available.
            show_plots (bool, optional): Show plot windows when visualization is generated.

        Raises:
            RuntimeError: If no agent has been created or if the collector
                returns zero steps.
        """

        # Load checkpoint
        # self._agent.policy = PPOPolicy.load("PySimEnv_PPO_policy", device="cpu")

        if not self._agent:
            raise RuntimeError("No agent has been created yet!")

        self.agent.set_policy_eval()
        raw_env = self.get_env()
        if hasattr(raw_env, "envs"):    # VecEnv wrapping
            raw_env = raw_env.envs[0]
        elif hasattr(raw_env, "env"):   # gym Wrapper
            raw_env = raw_env.env

        # Flip on eval-time realism overrides (variable TS radii, lateral
        # lane spawn, late head-on dodge). All overrides are opt-in via
        # env attributes that default to no-op in training.
        if hasattr(raw_env, "set_evaluation_parameters"):
            raw_env.set_evaluation_parameters()
        for sub in getattr(raw_env, "sub_environments", {}).values():
            if hasattr(sub, "set_evaluation_parameters"):
                sub.set_evaluation_parameters()

        is_lunar_lander = isinstance(raw_env, LunarLander)

        collector = EpisodeCollector(self._agent, raw_env, self._profile, airsim_bridge=airsim_bridge)
        run_data = collector.collect(num_episodes=num_episodes, live_animation=live_animation if not is_lunar_lander else True)

        if not is_lunar_lander:
            summaries = {}
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
                summaries[env_name] = {
                    "average_reward": m.avg_reward,
                    "best_episode_reward": m.best_reward,
                    "worst_episode_reward": m.worst_reward,
                    "mu": m.mu,
                    "Mu": m.Mu,
                    "mTEI": m.mTEI,
                    "MTE": m.MTE,
                    "mRA": m.mRA,
                    "MRA": m.MRA,
                    "mRAC": m.mRAC,
                    "vRA": m.vRA,
                    "sCF": m.sCF,
                    "mT": m.mT,
                    "MT": m.MT,
                    "mTC": m.mTC,
                    "mHE": m.mHE,
                }
                training_name = getattr(ship_env, "training_name", sid)
                print(f"\n{training_name} evaluation completed!")
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
                print(f"Variance of Rudder Angle (vRA): {m.vRA:.4f}")
                print(f"Steering Sign-Change Frequency (sCF): {m.sCF:.4f}")
                print(f"Mean Thrust (mT): {m.mT:.3f}")
                print(f"Maximum Thrust (MT): {m.MT:.3f}")
                print(f"Mean Thrust Change (mTC): {m.mTC:.4f}")
                print(f"Mean LOS Guidance Heading Error (mHE): {m.mHE:.3f} rad")
                print("Collision rate per policy (over valid-outcome encounters: collided OR resolved):")
                for policy in sorted(m.total_encounters_per_policy):
                    enc_n = m.encounters_per_policy.get(policy, 0)
                    total_n = m.total_encounters_per_policy[policy]
                    col_n = m.collisions_per_policy.get(policy, 0)
                    rate = (col_n / enc_n * 100) if enc_n else 0.0
                    print(
                        f"  {policy:<16s} {rate:6.2f}% "
                        f"({col_n}/{enc_n} valid; {total_n} total starts)"
                    )
                print("Encounter outcome breakdown per policy (all encounter starts):")
                cat_order = (
                    "collided",
                    "resolved",
                    "passed_not_tracked_lost",
                    "tracked_lost_not_passed",
                    "neither",
                )
                for policy in sorted(m.total_encounters_per_policy):
                    buckets = m.encounter_outcomes_per_policy.get(policy, {})
                    total_n = m.total_encounters_per_policy[policy]
                    parts = [f"{cat}={buckets.get(cat, 0)}" for cat in cat_order]
                    print(f"  {policy:<16s} total={total_n}  " + "  ".join(parts))
                print("CRI per policy (mean / median / max over valid encounters):")
                for policy in sorted(m.total_encounters_per_policy):
                    n_valid = m.valid_encounters_per_policy.get(policy, 0)
                    mean_cri = m.mean_cri_per_policy.get(policy, 0.0)
                    median_cri = m.median_cri_per_policy.get(policy, 0.0)
                    max_cri = m.max_cri_per_policy.get(policy, 0.0)
                    print(
                        f"  {policy:<16s} mean={mean_cri:.3f}  "
                        f"median={median_cri:.3f}  max={max_cri:.3f}  "
                        f"(n={n_valid})"
                    )
                # Baseline CRI floor - single global value, useful for
                # reading the per-policy CRI numbers against. Theoretical
                # baseline uses the CRI formula at DCPA=encounter_range,
                # TCPA<=0 (parting); the TCPA term collapses to 0 so only
                # the DCPA half remains.
                enc_r = float(getattr(ship_env, "encounter_range", 0.0))
                cri_D = float(getattr(ship_env, "cri_D", 1.0))
                cri_alpha1 = float(getattr(ship_env, "cri_alpha1", -1.0))
                cri_omega1 = float(getattr(ship_env, "cri_omega1", 0.5))
                theoretical = cri_omega1 * math.exp(cri_alpha1 * enc_r / max(cri_D, 1e-12))
                m.baseline_cri_theoretical = theoretical
                n_baseline = m.baseline_cri_sample_count
                print(
                    "Baseline CRI (lowest-achievable reference, across all policies):"
                )
                print(
                    f"  empirical: mean(per-encounter min CRI)={m.baseline_cri_empirical_mean:.3f}  "
                    f"min(observed)={m.baseline_cri_empirical_min:.3f}  (n={n_baseline})"
                )
                print(
                    f"  theoretical: CRI at DCPA=encounter_range ({enc_r:.0f}m), "
                    f"TCPA parting = {theoretical:.3f}"
                )
                print("Cross-track error per policy [m] (mean / median / max over valid encounters):")
                for policy in sorted(m.total_encounters_per_policy):
                    n_valid = m.valid_encounters_per_policy.get(policy, 0)
                    mean_cte = m.mean_cte_per_policy.get(policy, 0.0)
                    median_cte = m.median_cte_per_policy.get(policy, 0.0)
                    max_cte = m.max_cte_per_policy.get(policy, 0.0)
                    print(
                        f"  {policy:<16s} mean={mean_cte:.2f}m  "
                        f"median={median_cte:.2f}m  max={max_cte:.2f}m  "
                        f"(n={n_valid})"
                    )
                print("Time to first action per policy [s] (|rudder| > 10 deg, over valid encounters):")
                for policy in sorted(m.total_encounters_per_policy):
                    n_reactive = m.reactive_encounters_per_policy.get(policy, 0)
                    n_non_reactive = m.non_reactive_encounters_per_policy.get(policy, 0)
                    mean_tfa = m.mean_tfa_per_policy.get(policy, 0.0)
                    median_tfa = m.median_tfa_per_policy.get(policy, 0.0)
                    print(
                        f"  {policy:<16s} mean={mean_tfa:.2f}s  "
                        f"median={median_tfa:.2f}s  "
                        f"(n_reactive={n_reactive}, n_non_reactive={n_non_reactive})"
                    )
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

                # Scenario classification diagnostics - per encounter
                # (one record captured at encounter entry).
                enc_rate = (
                    m.encounter_misclassification_count / m.num_encounters * 100
                    if m.num_encounters else 0.0
                )
                print(
                    f"Scenario misclassification - "
                    f"{m.encounter_misclassification_count}/{m.num_encounters} encounters "
                    f"({enc_rate:.2f}%)"
                )
                if m.encounter_confusion:
                    print("Encounter confusion (gt -> pred: count):")
                    for (gt, pred), cnt in sorted(m.encounter_confusion.items()):
                        tag = "" if gt == pred else "  [MISMATCH]"
                        print(f"  {gt} -> {pred}: {cnt}{tag}")

            save_dir = self.agent.model_dir or "."
            if not save_dir:
                save_dir = "models"
            os.makedirs(save_dir, exist_ok=True)

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
                save_path=f"{save_dir}/image.png",
                show_plots=show_plots,
            )

                # PlotlyRenderer.render(
                #     run_data.ships, metrics,
                # )

            # Cleanup the environment
            fh_sim = getattr(self.get_env(), "fh_sim", None)
            if fh_sim is not None:
                fh_sim.dispose()

            if return_summary:
                return summaries

        return None

    def get_env(self):
        """
        Retrieve the environment associated with the agent.

        Returns:
            The environment object.
        """

        return self._agent.get_env()
