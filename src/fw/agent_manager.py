import os

from pathlib import Path
from fw.config_profile import VesselProfile
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
from fw.config import PPO_POLICY_NAME, PPO2_POLICY_NAME, MULTI_PPO_POLICY_NAME, PolicyStrategy
from fw.profile_applier import apply_profile


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
                print(f"Variance of Rudder Angle (vRA): {m.vRA:.4f}")
                print(f"Steering Sign-Change Frequency (sCF): {m.sCF:.4f}")
                print(f"Mean Thrust (mT): {m.mT:.3f}")
                print(f"Maximum Thrust (MT): {m.MT:.3f}")
                print(f"Mean Thrust Change (mTC): {m.mTC:.4f}")
                print(f"Mean LOS Guidance Heading Error (mHE): {m.mHE:.3f} rad")


            save_dir = self.agent.model_dir or "."
            if not save_dir:
                save_dir = "models"
            os.makedirs(save_dir, exist_ok=True)

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
