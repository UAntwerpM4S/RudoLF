import os
import math
import logging

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
from fw.common.tools.fh_sim_utils import fh_sim_cleanup
from airsim.communication.airsim_bridge import AirSimBridge

from fw.policies.base_agent import BaseAgent
from fw.policies.ppo2_agent import PPO2Agent
from fw.policies.multi_ppo_agent import MultiPPOAgent
from fw.policies.single_ppo_agent import SinglePPOAgent
from fw.policies.multi_policy_single_agent import MultiPolicySingleAgent
from fw.collision_avoidance.scenario_classifier import SCENARIO_LABELS, NUM_SCENARIOS
from fw.configuration.config import PPO_POLICY_NAME, PPO2_POLICY_NAME, MULTI_PPO_POLICY_NAME, PolicyStrategy, MULTI_POLICY_SINGLE_NAME

logger = logging.getLogger(__name__)


class AgentManager:
    """
    Creates agents for environments, trains them and evaluates/visualizes them.
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
        The agent type (policy name).
        """

        return self._agent_type

    @property
    def profile(self) -> "VesselProfile":
        """
        The active VesselProfile.
        """

        return self._profile

    @property
    def agent(self) -> BaseAgent:
        """
        The current agent (None before `set_environment`).
        """

        return self._agent

    def _build_single_agent(self, env):
        """
        Create a SinglePPOAgent for `env`, or for its only sub-environment.

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

    def _build_multi_agent(self, env: BaseEnv):
        """
        Create a MultiPPOAgent with one SinglePPOAgent per sub-environment.

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
        Create a MultiPolicySingleAgent (one policy per COLAV scenario).

        Args:
            env: A single PySimColavEnv with `include_scenario_one_hot=True`
                (the one-hot suffix routes steps to policies), or a wrapper with
                exactly one sub-environment.

        Returns:
            MultiPolicySingleAgent: Initialized multi-policy agent.

        Raises:
            RuntimeError: If more than one sub-environment is provided.
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
            self._profile,
            environment=target_env,
            num_policies=NUM_SCENARIOS,
            policy_labels=list(SCENARIO_LABELS),
            **self._hyperparameters,
        )

        # Reuse the PySimEnv path-following policy as a frozen warm start.  Its
        # twelve-feature input is a strict prefix of the COLAV observation;
        # MultiPolicySingleAgent maps slot 0 to that prefix before inference.
        pretrained_path_policy = Path(__file__).resolve().parents[2] / "models" / "PySimEnv_PPO_policy.zip"
        if not pretrained_path_policy.is_file():
            raise FileNotFoundError(
                f"Frozen path-following policy not found: {pretrained_path_policy}"
            )
        pretrained_dir = agent.model_dir
        pretrained_specs = {
            0: {"file": str(pretrained_path_policy), "frozen": True},
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
            policy_strategy (PolicyStrategy, optional): Reset the policy, reuse
                the current one, or reuse another environment's policy.
            other_env_name (str, optional): Environment whose policy to reuse.
            profile (VesselProfile, optional): Replaces the active profile
                (used by the pipeline scheduler per block); None keeps it.
            hyperparameters (dict, optional): Replace the active hyperparameters;
                None keeps them.

        If a policy file to reuse is missing or incompatible, a warning is
        logged and the fresh (reset) policy is kept.

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

        # Release resources of the replaced agent (e.g. PPO2Agent's SB3 VecEnv).
        if self._agent is not None and callable(getattr(self._agent, "close", None)):
            try:
                self._agent.close()
            except Exception:
                logger.exception("Closing the previous agent failed; continuing.")

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

        # Propagate profile env settings (e.g. enable_smoothing).
        apply_profile(self._profile, env=env)

        if policy_file_name:
            full_policy_file_name = f"{policy_file_name}.zip"
            if self.agent.model_dir:
                full_policy_file_name = os.path.join(self.agent.model_dir, full_policy_file_name)

            if Path(full_policy_file_name).exists():
                logger.info("Reusing %s.", policy_file_name)
                try:
                    self._agent.load_policy(policy_file_name)
                except (ValueError, RuntimeError) as e:
                    logger.warning("[Fallback] Incompatible model: %s", e)
                    logger.warning("[Fallback] Reverting to RESET_POLICY for %s.", self._agent_type)
            else:
                logger.warning("[Fallback] The policy file %s does not exist.", policy_file_name)
                logger.warning("[Fallback] Reverting to RESET_POLICY for %s.", self._agent_type)

    def train(self, stop_condition: StopCondition, dispose_env: bool = True) -> None:
        """
        Train the agent in the assigned environment.

        Args:
            stop_condition (StopCondition): The condition that determines when
                training should stop.
            dispose_env (bool, optional): Dispose the FH simulator bridge (if
                any) afterward. Pass False when `visualize_trained_agent()`
                follows on the same env (it disposes itself), to avoid a
                needless math-model reinitialization.

        Raises:
            RuntimeError: If no agent has been created yet.
        """

        if not self._agent:
            raise RuntimeError("No agent has been created yet!")

        # fh_sim_cleanup disposes also when learn() raises.
        with fh_sim_cleanup(self.get_env, dispose=dispose_env):
            self._agent.learn(stop_condition=stop_condition)

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
        Evaluate the trained agent, log metrics and save/show plots.

        Args:
            num_episodes (int, optional): Number of evaluation episodes.
            live_animation (bool, optional): Animate the episodes live.
            env_color_index (int, optional): Environment color index (currently unused).
            airsim_bridge (AirSimBridge, optional): Bridge for mirroring in Cosys-AirSim.
            return_summary (bool, optional): Return the per-environment metric summary.
            show_plots (bool, optional): Show plot windows.

        Returns:
            dict | None: Metric summary per environment name if `return_summary`
            (not for LunarLander), else None.

        Raises:
            RuntimeError: If no agent has been created or if the collector
                returns zero steps.
        """

        # Load checkpoint
        # self._agent.policy = PPOPolicy.load("PySimEnv_PPO_policy", device="cpu")

        if not self._agent:
            raise RuntimeError("No agent has been created yet!")

        with fh_sim_cleanup(self.get_env):
            self.agent.set_policy_eval()
            raw_env = self.get_env()
            if hasattr(raw_env, "envs"):    # VecEnv wrapping
                raw_env = raw_env.envs[0]
            elif hasattr(raw_env, "env"):   # gym Wrapper
                raw_env = raw_env.env

            # Enable eval-only realism (variable TS radii, lateral lane spawn,
            # late head-on dodge); these are off during training.
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
                    logger.info("%s evaluation completed!", training_name)
                    logger.info(f"Average reward: {m.avg_reward:.2f}")
                    logger.info(f"Best episode reward: {m.best_reward:.2f}")
                    logger.info(f"Worst episode reward: {m.worst_reward:.2f}")
                    logger.info(f"Mean Surge Velocity (mu): {m.mu:.2f}")
                    logger.info(f"Maximum Surge Velocity (Mu): {m.Mu:.2f}")
                    logger.info(f"Mean Track Error Integral (mTEI): {m.mTEI:.2f}")
                    logger.info(f"Maximum Track Error (MTE): {m.MTE:.2f}")
                    logger.info(f"Mean Rudder Angle (mRA): {m.mRA:.3f}")
                    logger.info(f"Maximum Rudder Angle (MRA): {m.MRA:.3f}")
                    logger.info(f"Mean Rudder Angle Change (mRAC): {m.mRAC:.4f}")
                    logger.info(f"Variance of Rudder Angle (vRA): {m.vRA:.4f}")
                    logger.info(f"Steering Sign-Change Frequency (sCF): {m.sCF:.4f}")
                    logger.info(f"Mean Thrust (mT): {m.mT:.3f}")
                    logger.info(f"Maximum Thrust (MT): {m.MT:.3f}")
                    logger.info(f"Mean Thrust Change (mTC): {m.mTC:.4f}")
                    logger.info(f"Mean LOS Guidance Heading Error (mHE): {m.mHE:.3f} rad")
                    logger.info("Collision rate per policy (over valid-outcome encounters: collided OR resolved):")
                    for policy in sorted(m.total_encounters_per_policy):
                        enc_n = m.encounters_per_policy.get(policy, 0)
                        total_n = m.total_encounters_per_policy[policy]
                        col_n = m.collisions_per_policy.get(policy, 0)
                        rate = (col_n / enc_n * 100) if enc_n else 0.0
                        logger.info(
                            f"  {policy:<16s} {rate:6.2f}% "
                            f"({col_n}/{enc_n} valid; {total_n} total starts)"
                        )
                    logger.info("Encounter outcome breakdown per policy (all encounter starts):")
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
                        logger.info(f"  {policy:<16s} total={total_n}  " + "  ".join(parts))
                    logger.info("CRI per policy (mean / median / max over valid encounters):")
                    for policy in sorted(m.total_encounters_per_policy):
                        n_valid = m.valid_encounters_per_policy.get(policy, 0)
                        mean_cri = m.mean_cri_per_policy.get(policy, 0.0)
                        median_cri = m.median_cri_per_policy.get(policy, 0.0)
                        max_cri = m.max_cri_per_policy.get(policy, 0.0)
                        logger.info(
                            f"  {policy:<16s} mean={mean_cri:.3f}  "
                            f"median={median_cri:.3f}  max={max_cri:.3f}  "
                            f"(n={n_valid})"
                        )
                    # Baseline CRI as reference for the per-policy values. The
                    # theoretical value is CRI at DCPA=encounter_range with TCPA<=0
                    # (parting), where only the DCPA term remains.
                    enc_r = float(getattr(ship_env, "encounter_range", 0.0))
                    cri_D = float(getattr(ship_env, "cri_D", 1.0))
                    cri_alpha1 = float(getattr(ship_env, "cri_alpha1", -1.0))
                    cri_omega1 = float(getattr(ship_env, "cri_omega1", 0.5))
                    theoretical = cri_omega1 * math.exp(cri_alpha1 * enc_r / max(cri_D, 1e-12))
                    m.baseline_cri_theoretical = theoretical
                    n_baseline = m.baseline_cri_sample_count
                    logger.info(
                        "Baseline CRI (lowest-achievable reference, across all policies):"
                    )
                    logger.info(
                        f"  empirical: mean(per-encounter min CRI)={m.baseline_cri_empirical_mean:.3f}  "
                        f"min(observed)={m.baseline_cri_empirical_min:.3f}  (n={n_baseline})"
                    )
                    logger.info(
                        f"  theoretical: CRI at DCPA=encounter_range ({enc_r:.0f}m), "
                        f"TCPA parting = {theoretical:.3f}"
                    )
                    logger.info("Cross-track error per policy [m] (mean / median / max over valid encounters):")
                    for policy in sorted(m.total_encounters_per_policy):
                        n_valid = m.valid_encounters_per_policy.get(policy, 0)
                        mean_cte = m.mean_cte_per_policy.get(policy, 0.0)
                        median_cte = m.median_cte_per_policy.get(policy, 0.0)
                        max_cte = m.max_cte_per_policy.get(policy, 0.0)
                        logger.info(
                            f"  {policy:<16s} mean={mean_cte:.2f}m  "
                            f"median={median_cte:.2f}m  max={max_cte:.2f}m  "
                            f"(n={n_valid})"
                        )
                    logger.info("Time to first action per policy [s] (|rudder| > 10 deg, over valid encounters):")
                    for policy in sorted(m.total_encounters_per_policy):
                        n_reactive = m.reactive_encounters_per_policy.get(policy, 0)
                        n_non_reactive = m.non_reactive_encounters_per_policy.get(policy, 0)
                        mean_tfa = m.mean_tfa_per_policy.get(policy, 0.0)
                        median_tfa = m.median_tfa_per_policy.get(policy, 0.0)
                        logger.info(
                            f"  {policy:<16s} mean={mean_tfa:.2f}s  "
                            f"median={median_tfa:.2f}s  "
                            f"(n_reactive={n_reactive}, n_non_reactive={n_non_reactive})"
                        )
                    n_ep = m.num_episodes
                    out_rate = (m.collisions_outside_encounters / n_ep * 100) if n_ep else 0.0
                    logger.info(f"  outside-encounter {out_rate:6.2f}% ({m.collisions_outside_encounters}/{n_ep} episodes)")
                    in_enc = m.collision_count - m.collisions_outside_encounters
                    total_denom = m.num_encounters + n_ep
                    cr_pct = (m.collision_count / total_denom * 100) if total_denom else 0.0
                    logger.info(
                        f"Total collision rate (CR): {cr_pct:.2f}% "
                        f"({m.collision_count} = {in_enc} in {m.num_encounters} encounters "
                        f"+ {m.collisions_outside_encounters} outside / {n_ep} episodes)"
                    )
                    logger.info(f"Collision per type: {m.collision_type_count}")
                    logger.info(f"Collision per side: {m.collision_side_count}")
                    logger.info(f"COLREG violation count: {m.colreg_violation_count}")

                    # Scenario classification diagnostics (one record per encounter).
                    enc_rate = (
                        m.encounter_misclassification_count / m.num_encounters * 100
                        if m.num_encounters else 0.0
                    )
                    logger.info(
                        f"Scenario misclassification - "
                        f"{m.encounter_misclassification_count}/{m.num_encounters} encounters "
                        f"({enc_rate:.2f}%)"
                    )
                    if m.encounter_confusion:
                        logger.info("Encounter confusion (gt -> pred: count):")
                        for (gt, pred), cnt in sorted(m.encounter_confusion.items()):
                            tag = "" if gt == pred else "  [MISMATCH]"
                            logger.info(f"  {gt} -> {pred}: {cnt}{tag}")

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
